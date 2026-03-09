"""
Observation construction for the PalletTask environment.

Builds the flattened observation vector from heightmap, buffer state,
box dimensions, payload, constraints, and proprioception.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pallet_rl.envs.pallet_task import PalletTask


def build_observations(env: PalletTask) -> dict[str, torch.Tensor]:
    """
    Construct observations (GPU-only, no numpy).

    Returns:
        dict: ``{"policy": (N, obs_dim), "critic": (N, obs_dim)}``
    """
    cfg = env.cfg
    n = env.num_envs
    device = torch.device(env._device)

    # ------------------------------------------------------------------
    # 1. Box poses from scene
    # ------------------------------------------------------------------
    M = cfg.max_boxes
    K = cfg.num_boxes

    if "boxes" in env.scene.keys():
        from pallet_rl.utils.quaternions import wxyz_to_xyzw

        boxes_data = env.scene["boxes"].data
        all_pos = boxes_data.object_pos_w.reshape(-1, 3)
        all_rot_wxyz = boxes_data.object_quat_w.reshape(-1, 4)

        box_pos = all_pos.view(n, M, 3)
        box_rot = wxyz_to_xyzw(all_rot_wxyz).view(n, M, 4)
    else:
        box_pos = torch.zeros(n, M, 3, device=device)
        box_rot = torch.zeros(n, M, 4, device=device)
        box_rot[:, :, 0] = 1.0

    pallet_pos = torch.zeros(n, 3, device=device)

    # ------------------------------------------------------------------
    # 2. Generate heightmap
    # ------------------------------------------------------------------
    heightmap = _generate_heightmap(env, box_pos, box_rot, pallet_pos, n, device)

    # ------------------------------------------------------------------
    # 3. Normalize/flatten heightmap and cache raw version
    # ------------------------------------------------------------------
    heightmap_norm = heightmap / cfg.max_height
    heightmap_flat = heightmap_norm.view(n, -1)
    env._last_heightmap = heightmap  # (N, H, W) in meters

    # ------------------------------------------------------------------
    # 4. Buffer state
    # ------------------------------------------------------------------
    buffer_flat = env.buffer_state.view(n, -1)

    # ------------------------------------------------------------------
    # 5. Current box dimensions
    # ------------------------------------------------------------------
    idx = env.box_idx.clamp(0, cfg.max_boxes - 1)
    env_idx = torch.arange(n, device=device)
    current_dims = env.box_dims[env_idx, idx]

    # ------------------------------------------------------------------
    # 6. Payload and mass observations
    # ------------------------------------------------------------------
    payload_norm = (env.payload_kg / cfg.max_payload_kg).unsqueeze(-1)
    max_box_mass = cfg.base_box_mass_kg + cfg.box_mass_variance
    current_box_mass = env.box_mass_kg[env_idx, idx]
    current_mass_norm = (current_box_mass / max_box_mass).unsqueeze(-1)

    # ------------------------------------------------------------------
    # 7. Constraint observations (constant now, ready for domain rand)
    # ------------------------------------------------------------------
    max_payload_norm = torch.full((n, 1), cfg.max_payload_kg / 1000.0, device=device)
    max_stack_height_norm = torch.full((n, 1), cfg.max_stack_height / 3.0, device=device)

    # ------------------------------------------------------------------
    # 8. Proprioception placeholder
    # ------------------------------------------------------------------
    proprio = torch.zeros(n, cfg.robot_state_dim, device=device)

    # ------------------------------------------------------------------
    # 9. Concatenate
    # ------------------------------------------------------------------
    obs = torch.cat([
        heightmap_flat,
        buffer_flat,
        current_dims,
        payload_norm,
        current_mass_norm,
        max_payload_norm,
        max_stack_height_norm,
        proprio,
    ], dim=-1)

    # Shape assertion
    if cfg.num_observations is None:
        cfg.num_observations = int(obs.shape[-1])
    expected_obs_dim = int(cfg.num_observations)
    assert obs.shape == (n, expected_obs_dim), \
        f"Obs shape {obs.shape} != expected ({n}, {expected_obs_dim})"
    assert obs.device == device, \
        f"Obs device {obs.device} != expected {device}"

    return {"policy": obs, "critic": obs}


def _generate_heightmap(
    env: PalletTask,
    box_pos: torch.Tensor,
    box_rot: torch.Tensor,
    pallet_pos: torch.Tensor,
    n: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate a heightmap using the configured backend pipeline."""
    # If the generator was intentionally skipped (e.g., mockup_mode RGB),
    # return a safe zero tensor to satisfy observation dimensions without Warp.
    if env.heightmap_gen is None and env.cfg.heightmap_source != "depth_camera":
        map_h, map_w = env.cfg.map_shape
        return torch.zeros(n, map_h, map_w, dtype=torch.float32, device=device)

    return env._heightmap_backend.generate(env)
