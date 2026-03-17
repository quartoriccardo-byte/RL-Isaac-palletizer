"""
Depth Camera → Heightmap Conversion Pipeline (GPU, Batched)

Converts depth-camera images into 2D heightmap grids identical in shape
to the Warp-based rasterizer output, so the rest of the observation
pipeline is agnostic to the heightmap source.

All operations stay on GPU. No CPU copies.

Pipeline:
    depth (N, H_cam, W_cam) → noise → unproject (Camera Frame) → transform (World Frame)
    → crop to pallet region → rasterize (scatter_reduce amax) → (N, H_map, W_map)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DepthHeightmapCfg:
    """Configuration for depth-to-heightmap conversion."""

    # Camera intrinsics
    cam_height: int = 160
    cam_width: int = 240
    fov_deg: float = 40.0

    # Sensor height above ground (m)
    sensor_height_m: float = 3.0

    # Output heightmap shape
    map_h: int = 160
    map_w: int = 240

    # Crop bounds (world XY, meters)
    crop_x: tuple[float, float] = (-0.65, 0.65)
    crop_y: tuple[float, float] = (-0.45, 0.45)

    # Noise model
    noise_enable: bool = True
    noise_sigma_m: float = 0.003
    noise_scale: float = 0.7  # underestimate factor
    noise_quantization_m: float = 0.002
    noise_dropout_prob: float = 0.001

    # Diagnostics
    debug_stats: bool = False
    debug_save_dir: str = ""


class DepthHeightmapConverter:
    """
    Converts batched depth images to heightmap grids (GPU, batched).

    Usage::

        converter = DepthHeightmapConverter(cfg, device="cuda")
        heightmap = converter(depth_images, cam_pos, cam_quat)
        # heightmap: (N, map_h, map_w)
    """

    def __init__(self, cfg: DepthHeightmapCfg, device: str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device)

        # =====================================================================
        # Precompute pixel-ray directions in camera frame
        # =====================================================================
        # Pinhole model: fov_deg is the vertical field of view
        fy = cfg.cam_height / (2.0 * math.tan(math.radians(cfg.fov_deg / 2.0)))
        fx = fy  # square pixels
        cx = cfg.cam_width / 2.0
        cy = cfg.cam_height / 2.0

        # Pixel coordinates grid: (H, W)
        u = torch.arange(cfg.cam_width, dtype=torch.float32, device=self.device)
        v = torch.arange(cfg.cam_height, dtype=torch.float32, device=self.device)
        vv, uu = torch.meshgrid(v, u, indexing="ij")  # (H, W)

        # Ray directions in camera frame (OpenCV convention: +Z forward)
        self._ray_x = ((uu - cx) / fx).reshape(-1)  # (H*W,)
        self._ray_y = ((vv - cy) / fy).reshape(-1)  # (H*W,)
        # z = 1.0 for all rays (will be scaled by depth)

        # Crop-to-grid mapping constants
        self._crop_x_min = cfg.crop_x[0]
        self._crop_x_max = cfg.crop_x[1]
        self._crop_y_min = cfg.crop_y[0]
        self._crop_y_max = cfg.crop_y[1]
        self._crop_x_range = cfg.crop_x[1] - cfg.crop_x[0]
        self._crop_y_range = cfg.crop_y[1] - cfg.crop_y[0]

        # Step counter for decimation
        self._step_count = 0
        self._cached_heightmap: torch.Tensor | None = None

    def apply_noise(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Apply realistic depth sensor noise (underestimated).

        Args:
            depth: Raw depth values (N, H, W)

        Returns:
            Noisy depth values (N, H, W)
        """
        if not self.cfg.noise_enable:
            return depth

        # Gaussian noise (underestimated)
        sigma = self.cfg.noise_sigma_m * self.cfg.noise_scale
        noise = torch.randn_like(depth) * sigma
        depth = depth + noise

        # Quantization (simulate discrete sensor ADC)
        if self.cfg.noise_quantization_m > 0:
            q = self.cfg.noise_quantization_m
            depth = torch.round(depth / q) * q

        # Dropout (simulate missing depth readings)
        if self.cfg.noise_dropout_prob > 0:
            mask = torch.rand_like(depth) > self.cfg.noise_dropout_prob
            depth = depth * mask  # dropout → 0.0 (will be filtered as invalid)

        return depth

    @staticmethod
    def _quat_to_rotation_matrix(quat_wxyz: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternion (w, x, y, z) to 3×3 rotation matrix.

        Args:
            quat_wxyz: (N, 4) quaternions in wxyz order

        Returns:
            R: (N, 3, 3) rotation matrices
        """
        w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]

        R = torch.zeros(quat_wxyz.shape[0], 3, 3, device=quat_wxyz.device, dtype=quat_wxyz.dtype)

        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - w * z)
        R[:, 0, 2] = 2 * (x * z + w * y)
        R[:, 1, 0] = 2 * (x * y + w * z)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - w * x)
        R[:, 2, 0] = 2 * (x * z - w * y)
        R[:, 2, 1] = 2 * (y * z + w * x)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)

        return R

    def depth_to_heightmap(
        self,
        depth: torch.Tensor,
        cam_pos: torch.Tensor,
        cam_quat_wxyz: torch.Tensor,
        frame_idx: int = -1,
        save_debug: bool = False,
    ) -> torch.Tensor:
        """
        Convert depth images to heightmap grids.

        Args:
            depth: Distance-to-image-plane depth (N, H_cam, W_cam)
            cam_pos: Camera world position (N, 3)
            cam_quat_wxyz: Camera world orientation (N, 4) in wxyz

        Returns:
            heightmap: (N, map_h, map_w) in meters above ground
        """
        N = depth.shape[0]
        H, W = self.cfg.cam_height, self.cfg.cam_width
        n_pix = H * W

        # 1. Apply noise
        depth = self.apply_noise(depth)

        # 2. Points in Camera Frame (OpenCV: +Z forward)
        # depth: (N, H, W)
        # self._ray_x: (H*W,)
        depth_flat = depth.reshape(N, -1)  # (N, H*W)
        
        # p_cam: (N, 3, H*W)
        p_cam = torch.stack([
            self._ray_x.unsqueeze(0).expand(N, -1) * depth_flat,
            self._ray_y.unsqueeze(0).expand(N, -1) * depth_flat,
            depth_flat
        ], dim=1)

        # 3. Transform to World Frame
        # R: (N, 3, 3), T: (N, 3, 1)
        R = self._quat_to_rotation_matrix(cam_quat_wxyz)
        T = cam_pos.unsqueeze(2)
        
        # p_world: (N, 3, H*W) = R * p_cam + T
        p_world = torch.bmm(R, p_cam) + T
        
        x_world = p_world[:, 0, :]
        y_world = p_world[:, 1, :]
        z_world_unclamped = p_world[:, 2, :]
        
        # 4. Explicit Background Masking
        # Depths close to the background plane are masked (if background plane exists).
        # We also enforce a hard cutoff below ground (Z < 0).
        z_world = torch.clamp(z_world_unclamped, min=0.0)
        
        # Background mask (depth-based or height-based)
        # If the camera is looking down from 3m, heights ~0m are background.
        bg_thresh = 0.005 # Threshold above ground to consider "not floor"
        bg_mask = (z_world < bg_thresh)

        valid = (
            (depth_flat > 0.01)       # discard dropout/invalid
            & (~bg_mask)              # discard floor/background
            & (x_world >= self._crop_x_min)
            & (x_world < self._crop_x_max)
            & (y_world >= self._crop_y_min)
            & (y_world < self._crop_y_max)
        )

        # 5. Rasterize: scatter max into grid cells
        map_h, map_w = self.cfg.map_h, self.cfg.map_w

        # Compute grid indices
        gx_raw = ((x_world - self._crop_x_min) / self._crop_x_range * map_w)
        gy_raw = ((y_world - self._crop_y_min) / self._crop_y_range * map_h)
        gx = gx_raw.long()
        gy = gy_raw.long()

        # Clamp to valid range
        gx = gx.clamp(0, map_w - 1)
        gy = gy.clamp(0, map_h - 1)

        # Flat cell index per pixel
        cell_idx = gy * map_w + gx  # (N, H*W)

        # Initialize heightmap to 0 (ground level)
        heightmap = torch.zeros(N, map_h * map_w, device=self.device)

        # For each env, scatter max
        # We batch by repeating env index: global_cell = env * (map_h * map_w) + cell
        env_offset = torch.arange(N, device=self.device).unsqueeze(1) * (map_h * map_w)
        global_cell = (cell_idx + env_offset).reshape(-1)  # (N*H*W,)
        z_flat = z_world.reshape(-1)  # (N*H*W,)
        valid_flat = valid.reshape(-1)

        # Only scatter valid points
        valid_idx = valid_flat.nonzero(as_tuple=True)[0]
        if valid_idx.numel() > 0:
            heightmap_flat = heightmap.reshape(-1)  # (N * map_h * map_w,)
            heightmap_flat.scatter_reduce_(
                0,
                global_cell[valid_idx],
                z_flat[valid_idx],
                reduce="amax",
                include_self=True,
            )

        heightmap = heightmap.reshape(N, map_h, map_w)

        # Diagnostics Logging
        if self.cfg.debug_stats and save_debug:
            print("\n[HMAP_DEBUG2] --- Depth to Heightmap Converter Diagnostics ---")
            print(f"[HMAP_DEBUG2] Frame {frame_idx} | Input Depth | Shape: {depth.shape}, Dtype: {depth.dtype}, Device: {depth.device}")
            print(f"[HMAP_DEBUG2] Input Depth | Min: {depth_flat.min():.4f}, Max: {depth_flat.max():.4f}, Mean: {depth_flat.mean():.4f}")
            
            print(f"[HMAP_DEBUG2] X_world     | Min: {x_world.min():.4f}, Max: {x_world.max():.4f}, Mean: {x_world.mean():.4f}")
            print(f"[HMAP_DEBUG2] Y_world     | Min: {y_world.min():.4f}, Max: {y_world.max():.4f}, Mean: {y_world.mean():.4f}")

            in_crop_mask = (
                (x_world >= self._crop_x_min) & (x_world < self._crop_x_max) & 
                (y_world >= self._crop_y_min) & (y_world < self._crop_y_max)
            )
            print(f"[HMAP_DEBUG2] Crop Valid  | Pixels in bounds: {in_crop_mask.sum().item()} / {in_crop_mask.numel()}")
            
            finite_mask = torch.isfinite(depth_flat)
            print(f"[HMAP_DEBUG2] Finite Px   | {finite_mask.sum().item()} / {depth_flat.numel()}")
            
            print(f"[HMAP_DEBUG2] Background  | Threshold: >= {bg_thresh:.3f}m | Pixels masked: {bg_mask.sum().item()} / {depth_flat.numel()}")
            
            print(f"[HMAP_DEBUG2] Final Valid | valid_flat sum: {valid_flat.sum().item()} / {valid_flat.numel()}")
            
            print(f"[HMAP_DEBUG2] Proj Map X  | gx_raw Min: {gx_raw.min():.1f}, Max: {gx_raw.max():.1f}")
            print(f"[HMAP_DEBUG2] Proj Map Y  | gy_raw Min: {gy_raw.min():.1f}, Max: {gy_raw.max():.1f}")
            
            print(f"[HMAP_DEBUG2] Sensor Z    | {sensor_height:.4f}m configured")
            
            # Print stats only on valid pixels if any exist
            if valid_idx.numel() > 0:
                z_valid = z_world_unclamped.reshape(-1)[valid_idx]
                print(f"[HMAP_DEBUG2] Z Valid Uncl| Min: {z_valid.min():.4f}, Max: {z_valid.max():.4f}, Mean: {z_valid.mean():.4f}")
            else:
                print(f"[HMAP_DEBUG2] Z Valid Uncl| NO VALID PIXELS TO MEASURE")

            print(f"[HMAP_DEBUG2] Z PRE-clamp | Min: {z_world_unclamped.min():.4f}, Max: {z_world_unclamped.max():.4f}, Mean: {z_world_unclamped.mean():.4f}")
            
            z_neg_count = (z_world_unclamped < 0).sum().item()
            print(f"[HMAP_DEBUG2] Z Negatives | {z_neg_count} pixels clamped to 0.0")
            print(f"[HMAP_DEBUG2] Z POST-clamp| Min: {z_world.min():.4f}, Max: {z_world.max():.4f}, Mean: {z_world.mean():.4f}")
            
            # Scatter stats
            num_nonzero_hmap = (heightmap > 0).sum().item()
            if num_nonzero_hmap > 0:
                print(f"[HMAP_DEBUG2] Scatted Out | Non-zero cells: {num_nonzero_hmap} | Min: {(heightmap[heightmap>0]).min():.4f}, Max: {heightmap.max():.4f}")
            else:
                print(f"[HMAP_DEBUG2] Scatted Out | ALL CELLS ARE ZERO")
                
            print("[HMAP_DEBUG2] ------------------------------------------------\n")
            
            # Save preclamp debug array (overwrite or sequence depending on run mode)
            import os
            import numpy as np
            if self.cfg.debug_save_dir:
                os.makedirs(self.cfg.debug_save_dir, exist_ok=True)
                z_flat_unclamped = z_world_unclamped.reshape(-1)
                hmap_preclamp = torch.zeros(N, map_h * map_w, device=self.device)
                hmap_preclamp_flat = hmap_preclamp.reshape(-1)
                if valid_idx.numel() > 0:
                    hmap_preclamp_flat.scatter_reduce_(
                        0, global_cell[valid_idx], z_flat_unclamped[valid_idx], reduce="amax", include_self=True
                    )
                hmap_preclamp = hmap_preclamp.reshape(N, map_h, map_w).cpu().numpy()[0]
                save_idx = frame_idx if frame_idx >= 0 else self._step_count
                save_path = os.path.join(self.cfg.debug_save_dir, f"hmap_preclamp_{save_idx:06d}.npy")
                np.save(save_path, hmap_preclamp.astype(np.float32))
                print(f"[HMAP_DEBUG2] Saved PRE-clamp heightmap tensor to {save_path}")

        return heightmap

    def __call__(
        self,
        depth: torch.Tensor,
        cam_pos: torch.Tensor,
        cam_quat_wxyz: torch.Tensor,
        decimation: int = 1,
    ) -> torch.Tensor:
        """
        Main entry point with optional decimation.

        Args:
            depth: (N, H_cam, W_cam) depth images
            cam_pos: (N, 3) camera world positions
            cam_quat_wxyz: (N, 4) camera orientations
            decimation: compute every N-th step, reuse cache otherwise

        Returns:
            heightmap: (N, map_h, map_w)
        """
        self._step_count += 1

        if decimation > 1 and self._cached_heightmap is not None:
            if (self._step_count - 1) % decimation != 0:
                return self._cached_heightmap

        hmap = self.depth_to_heightmap(depth, cam_pos, cam_quat_wxyz)
        self._cached_heightmap = hmap
        return hmap
