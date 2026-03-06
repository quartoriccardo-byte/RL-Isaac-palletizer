"""
Depth-camera heightmap backend adapter.

Wraps ``DepthHeightmapConverter`` to conform to ``BaseHeightmapBackend``.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from pallet_rl.envs.perception.base import BaseHeightmapBackend

if TYPE_CHECKING:
    from pallet_rl.envs.pallet_task import PalletTask


class DepthCameraBackend(BaseHeightmapBackend):
    """
    Heightmap generation from depth camera images.

    Uses the depth camera sensor attached to the scene, converts depth
    images to world-frame 3D points, and rasterizes into a heightmap grid.
    Supports noise injection and decimation for realistic sim-to-real.
    """

    def __init__(self):
        self._step_count = 0
        self._cached_heightmap: torch.Tensor | None = None

    @property
    def name(self) -> str:
        return "depth_camera"

    def generate(self, env: PalletTask) -> torch.Tensor:
        """Generate heightmap from depth camera sensor data."""
        cfg = env.cfg

        depth_cam = env.scene["depth_camera"]
        depth_data = depth_cam.data

        depth_img = depth_data.output["distance_to_image_plane"]
        if depth_img.dim() == 4 and depth_img.shape[-1] == 1:
            depth_img = depth_img.squeeze(-1)

        cam_pos = depth_data.pos_w
        cam_quat_wxyz = depth_data.quat_w_world

        # Decimation: reuse cached heightmap on non-update steps
        self._step_count += 1
        dec = cfg.depth_cam_decimation
        if dec > 1 and self._cached_heightmap is not None:
            if (self._step_count - 1) % dec != 0:
                return self._cached_heightmap

        heightmap = env._depth_converter.depth_to_heightmap(
            depth_img, cam_pos, cam_quat_wxyz
        )
        self._cached_heightmap = heightmap

        # Optional debug frame saving
        if cfg.depth_debug_save_frames:
            import os
            os.makedirs(cfg.depth_debug_save_dir, exist_ok=True)
            torch.save(
                {"depth": depth_img[0].cpu(), "heightmap": heightmap[0].cpu()},
                os.path.join(cfg.depth_debug_save_dir, f"frame_{self._step_count:06d}.pt"),
            )

        return heightmap

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear cached heightmap on reset."""
        if env_ids is None:
            self._cached_heightmap = None
            self._step_count = 0
