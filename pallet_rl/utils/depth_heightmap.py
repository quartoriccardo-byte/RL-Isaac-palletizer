"""
Depth Camera → Heightmap Conversion Pipeline (GPU, Batched)

Converts depth-camera images into 2D heightmap grids identical in shape
to the Warp-based rasterizer output, so the rest of the observation
pipeline is agnostic to the heightmap source.

All operations stay on GPU. No CPU copies.

Pipeline:
    depth (N, H_cam, W_cam) → noise → unproject to 3D → world transform
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

        # 2. Unproject to camera-frame 3D points
        depth_flat = depth.reshape(N, -1)  # (N, H*W)

        # Camera-frame coordinates: (N, H*W, 3)
        pts_cam = torch.stack([
            self._ray_x.unsqueeze(0).expand(N, -1) * depth_flat,  # X
            self._ray_y.unsqueeze(0).expand(N, -1) * depth_flat,  # Y
            depth_flat,                                             # Z (forward)
        ], dim=-1)  # (N, H*W, 3)

        # 3. Transform to world frame
        R = self._quat_to_rotation_matrix(cam_quat_wxyz)  # (N, 3, 3)
        # pts_world = R @ pts_cam^T + cam_pos
        pts_world = torch.bmm(pts_cam, R.transpose(1, 2))  # (N, H*W, 3)
        pts_world = pts_world + cam_pos.unsqueeze(1)  # broadcast add

        # 4. Filter by crop bounds and valid depth
        x_world = pts_world[:, :, 0]  # (N, H*W)
        y_world = pts_world[:, :, 1]
        z_world = pts_world[:, :, 2]

        valid = (
            (depth_flat > 0.01)  # discard dropout / invalid readings
            & (x_world >= self._crop_x_min)
            & (x_world < self._crop_x_max)
            & (y_world >= self._crop_y_min)
            & (y_world < self._crop_y_max)
        )  # (N, H*W) bool

        # 5. Rasterize: scatter max into grid cells
        map_h, map_w = self.cfg.map_h, self.cfg.map_w

        # Compute grid indices
        gx = ((x_world - self._crop_x_min) / self._crop_x_range * map_w).long()
        gy = ((y_world - self._crop_y_min) / self._crop_y_range * map_h).long()

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
