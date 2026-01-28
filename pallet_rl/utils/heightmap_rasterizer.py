"""
Warp-based GPU Heightmap Rasterizer

High-performance heightmap generation using NVIDIA Warp.
All operations stay on GPU - no numpy, no CPU transfers.

Input: PyTorch GPU tensors
Output: PyTorch GPU tensors
"""

from __future__ import annotations

import torch
import warp as wp

# -----------------------------------------------------------------------------
# Warp Initialization (Version-Safe Pattern)
# -----------------------------------------------------------------------------
# NOTE: wp.is_initialized() does NOT exist in Warp 1.11.x and newer versions.
# wp.init() is idempotent - calling it multiple times is safe and has no effect
# if Warp is already initialized. This pattern ensures compatibility across
# all Warp versions without raising AttributeError.
try:
    wp.init()
except Exception:
    # Warp may raise if already initialized in some edge cases; safe to ignore.
    pass


# =============================================================================
# Warp Kernel
# =============================================================================

@wp.kernel
def rasterize_heightmap_kernel(
    box_positions: wp.array(dtype=wp.vec3),      # (N_envs * max_boxes)
    box_orientations: wp.array(dtype=wp.quat),   # (N_envs * max_boxes)
    box_dimensions: wp.array(dtype=wp.vec3),     # (N_envs * max_boxes)
    pallet_positions: wp.array(dtype=wp.vec3),   # (N_envs)
    pallet_orientations: wp.array(dtype=wp.quat),# (N_envs)
    grid_map: wp.array(dtype=float, ndim=3),     # (N_envs, H, W)
    num_envs: int,
    max_boxes: int,
    grid_res: float,
    map_size_x: int,  # Height (rows)
    map_size_y: int   # Width (cols)
):
    """
    GPU-parallel heightmap rasterizer.
    
    Each thread handles one pixel across all boxes in one environment.
    Thread ID = env_id * (H * W) + pixel_index
    """
    tid = wp.tid()
    pixels_per_env = map_size_x * map_size_y
    
    # Determine environment and pixel coordinates
    env_id = tid // pixels_per_env
    pixel_idx = tid % pixels_per_env
    
    if env_id >= num_envs:
        return
    
    # Row (u) and column (v) indices
    u = pixel_idx // map_size_y  # Row [0..H-1]
    v = pixel_idx % map_size_y   # Col [0..W-1]
    
    # Grid physical dimensions
    grid_size_x = float(map_size_x) * grid_res
    grid_size_y = float(map_size_y) * grid_res
    
    # Pixel center in local grid frame
    px_local_x = (float(u) + 0.5) * grid_res - (grid_size_x * 0.5)
    px_local_y = (float(v) + 0.5) * grid_res - (grid_size_y * 0.5)
    
    # Transform pixel to world frame
    pallet_pos = pallet_positions[env_id]
    pallet_quat = pallet_orientations[env_id]
    
    p_offset_world = wp.quat_rotate(pallet_quat, wp.vec3(px_local_x, px_local_y, 0.0))
    p_world = pallet_pos + p_offset_world
    
    # Find maximum height at this pixel
    max_height = 0.0
    base_idx = env_id * max_boxes
    
    for i in range(max_boxes):
        idx = base_idx + i
        
        dims = box_dimensions[idx]
        if dims[0] <= 0.0:
            continue
        
        b_pos = box_positions[idx]
        b_quat = box_orientations[idx]
        
        # Transform pixel to box local frame
        diff = p_world - b_pos
        p_box_local = wp.quat_rotate_inv(b_quat, diff)
        
        # Check if pixel is inside box footprint
        half_l = dims[0] * 0.5
        half_w = dims[1] * 0.5
        
        if (p_box_local[0] >= -half_l and p_box_local[0] <= half_l and
            p_box_local[1] >= -half_w and p_box_local[1] <= half_w):
            
            # Pixel is inside - get box top height
            z_top = b_pos[2] + dims[2] * 0.5
            
            if z_top > max_height:
                max_height = z_top
    
    # Write result
    grid_map[env_id, u, v] = max_height


# =============================================================================
# Python Wrapper
# =============================================================================

class WarpHeightmapGenerator:
    """
    GPU-only heightmap generator using NVIDIA Warp.
    
    All inputs and outputs are PyTorch tensors on CUDA.
    No numpy arrays are used anywhere in the pipeline.
    
    Usage:
        gen = WarpHeightmapGenerator(device="cuda:0", ...)
        heightmap = gen.forward(box_pos, box_rot, box_dims, pallet_pos)
    """
    
    def __init__(
        self,
        device: str,
        num_envs: int,
        max_boxes: int,
        grid_res: float,
        map_size: tuple[int, int] | int | None = None,
        pallet_dims: tuple[float, float] | None = None,
        *,
        # Backwards-compatible alias used by some call sites / configs.
        # If both map_size and map_shape are provided they must agree.
        map_shape: tuple[int, int] | int | None = None,
    ):
        """
        Initialize the heightmap generator.
        
        Args:
            device: CUDA device string (e.g., "cuda:0")
            num_envs: Number of parallel environments
            max_boxes: Maximum boxes per environment
            grid_res: Grid resolution in meters
            map_size: (height, width) in pixels
            pallet_dims: (length, width) of pallet in meters
        """
        # Normalize device string
        if isinstance(device, torch.device):
            device = str(device)
        if "cuda" in device and ":" not in device:
            device = "cuda:0"
        
        self.device = device
        self.num_envs = num_envs
        self.max_boxes = max_boxes
        self.grid_res = grid_res
        
        # Resolve map size from alias arguments.
        # Prefer explicit map_size, but allow map_shape as a drop-in alias.
        if map_size is None and map_shape is None:
            raise ValueError("WarpHeightmapGenerator requires either map_size or map_shape.")

        if map_size is None:
            resolved = map_shape
        elif map_shape is None:
            resolved = map_size
        else:
            # Both provided – ensure they match to avoid silent shape bugs.
            if isinstance(map_size, int) and isinstance(map_shape, int):
                if map_size != map_shape:
                    raise ValueError(f"map_size ({map_size}) != map_shape ({map_shape})")
                resolved = map_size
            else:
                # Normalize to (H, W) then compare elementwise.
                if isinstance(map_size, int):
                    ms = (map_size, map_size)
                else:
                    ms = tuple(map_size)
                if isinstance(map_shape, int):
                    msh = (map_shape, map_shape)
                else:
                    msh = tuple(map_shape)
                if ms != msh:
                    raise ValueError(f"map_size {ms} != map_shape {msh}")
                resolved = ms

        if isinstance(resolved, int):
            self.map_size_x = resolved
            self.map_size_y = resolved
        else:
            self.map_size_x = int(resolved[0])
            self.map_size_y = int(resolved[1])

        assert self.map_size_x > 0 and self.map_size_y > 0, \
            f"Invalid heightmap size ({self.map_size_x}, {self.map_size_y})"

        self.pallet_dims = pallet_dims
        
        # Pre-allocate Warp output buffer
        self.grid_wp = wp.zeros(
            (num_envs, self.map_size_x, self.map_size_y),
            dtype=float,
            device=device
        )
    
    def forward(
        self,
        box_pos: torch.Tensor,
        box_rot: torch.Tensor,
        box_dims: torch.Tensor,
        pallet_pos: torch.Tensor,
        pallet_rot: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Generate heightmaps from box states.
        
        All inputs must be contiguous PyTorch GPU tensors.
        
        Args:
            box_pos: Box positions, shape (N*max_boxes, 3) or (N, max_boxes, 3)
            box_rot: Box quaternions **(x, y, z, w)**, shape (N*max_boxes, 4)
            box_dims: Box dimensions (L,W,H), shape (N*max_boxes, 3)
            pallet_pos: Pallet positions, shape (N, 3)
            pallet_rot: Pallet quaternions **(x, y, z, w)**, shape (N, 4), optional
            
        Returns:
            heightmap: Tensor of shape (N, H, W)
        """
        # Ensure inputs are contiguous
        box_pos = box_pos.contiguous().view(-1, 3)
        box_rot = box_rot.contiguous().view(-1, 4)
        box_dims = box_dims.contiguous().view(-1, 3)
        pallet_pos = pallet_pos.contiguous()

        # ------------------------------------------------------------------
        # Basic device / shape sanity checks (fail fast with clear messages)
        # ------------------------------------------------------------------
        expected_boxes = self.num_envs * self.max_boxes
        if box_pos.shape[0] != expected_boxes or box_dims.shape[0] != expected_boxes:
            raise ValueError(
                f"box_pos/box_dims first dim must be num_envs*max_boxes "
                f"({expected_boxes}), got "
                f"box_pos={box_pos.shape}, box_dims={box_dims.shape}"
            )
        if box_rot.shape[0] != expected_boxes or box_rot.shape[1] != 4:
            raise ValueError(
                f"box_rot must have shape (num_envs*max_boxes, 4), got {box_rot.shape}"
            )
        if pallet_pos.shape[0] != self.num_envs or pallet_pos.shape[1] != 3:
            raise ValueError(
                f"pallet_pos must have shape (num_envs, 3), got {pallet_pos.shape}"
            )

        # Device consistency between PyTorch and Warp.
        # We only support CUDA tensors here since the env is GPU-only.
        if not box_pos.is_cuda or not box_rot.is_cuda or not box_dims.is_cuda or not pallet_pos.is_cuda:
            raise RuntimeError(
                "WarpHeightmapGenerator expects all inputs on CUDA (GPU tensors)."
            )
        if not self.device.startswith("cuda"):
            raise RuntimeError(
                f"WarpHeightmapGenerator was constructed with device='{self.device}', "
                "but CUDA tensors were provided. Use a matching CUDA device string."
            )
        
        # Default pallet rotation (identity, x=y=z=0, w=1 in (x,y,z,w) convention)
        if pallet_rot is None:
            pallet_rot = torch.zeros(self.num_envs, 4, device=pallet_pos.device)
            pallet_rot[:, 3] = 1.0
        else:
            pallet_rot = pallet_rot.contiguous()
        
        # Clear output buffer
        self.grid_wp.zero_()
        
        # Convert to Warp arrays (zero-copy)
        box_pos_wp = wp.from_torch(box_pos, dtype=wp.vec3)
        box_rot_wp = wp.from_torch(box_rot, dtype=wp.quat)
        box_dims_wp = wp.from_torch(box_dims, dtype=wp.vec3)
        pallet_pos_wp = wp.from_torch(pallet_pos, dtype=wp.vec3)
        pallet_rot_wp = wp.from_torch(pallet_rot, dtype=wp.quat)
        
        # Launch kernel
        total_threads = self.num_envs * self.map_size_x * self.map_size_y
        
        wp.launch(
            kernel=rasterize_heightmap_kernel,
            dim=total_threads,
            inputs=[
                box_pos_wp,
                box_rot_wp,
                box_dims_wp,
                pallet_pos_wp,
                pallet_rot_wp,
                self.grid_wp,
                self.num_envs,
                self.max_boxes,
                self.grid_res,
                self.map_size_x,
                self.map_size_y
            ],
            device=self.device
        )
        
        # Convert back to PyTorch (zero-copy)
        return wp.to_torch(self.grid_wp)
