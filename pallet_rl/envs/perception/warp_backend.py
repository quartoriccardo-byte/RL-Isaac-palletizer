"""
Warp-based heightmap backend adapter.

Wraps ``WarpHeightmapGenerator`` to conform to ``BaseHeightmapBackend``.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from pallet_rl.envs.perception.base import BaseHeightmapBackend

if TYPE_CHECKING:
    from pallet_rl.envs.pallet_task import PalletTask


class WarpBackend(BaseHeightmapBackend):
    """
    Analytical heightmap generation using NVIDIA Warp GPU rasterizer.

    Computes heightmaps directly from box poses and dimensions —
    no cameras or depth images required. Best for training (fastest,
    no sensor noise, deterministic).
    """

    @property
    def name(self) -> str:
        return "warp"

    def generate(self, env: PalletTask) -> torch.Tensor:
        """Generate heightmap via analytical Warp rasterization."""
        from pallet_rl.utils.quaternions import wxyz_to_xyzw

        cfg = env.cfg
        n = env.num_envs
        device = torch.device(env._device)
        M = cfg.max_boxes

        # Read box poses from scene
        if "boxes" in env.scene.keys():
            boxes_data = env.scene["boxes"].data
            all_pos = boxes_data.object_pos_w.reshape(-1, 3)
            all_rot_wxyz = boxes_data.object_quat_w.reshape(-1, 4)
            box_pos = all_pos.view(n, M, 3)
            # Isaac (w,x,y,z) → Warp (x,y,z,w)
            box_rot = wxyz_to_xyzw(all_rot_wxyz).view(n, M, 4)
        else:
            box_pos = torch.zeros(n, M, 3, device=device)
            box_rot = torch.zeros(n, M, 4, device=device)
            box_rot[:, :, 0] = 1.0

        pallet_pos = torch.zeros(n, 3, device=device)

        # Mask inactive boxes
        box_indices = torch.arange(M, device=device).view(1, -1)
        active_mask = box_indices < env.box_idx.view(-1, 1)

        env._box_dims_for_hmap.copy_(env.box_dims)
        env._box_dims_for_hmap[~active_mask] = 0.0

        box_pos_for_hmap = box_pos.clone()
        box_pos_for_hmap[~active_mask] = env._inactive_box_pos

        return env.heightmap_gen.forward(
            box_pos_for_hmap.reshape(-1, 3),
            box_rot.reshape(-1, 4),
            env._box_dims_for_hmap.reshape(-1, 3),
            pallet_pos,
        )
