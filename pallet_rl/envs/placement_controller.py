"""
Placement controller: action decoding and physical placement for PalletTask.

Handles:
  - Continuous [-1, 1] → discrete/continuous sub-action decoding
  - Grid index → world position mapping
  - Height constraint validation
  - Pose writing to simulation via RigidObjectCollection API
  - Settling window arm for PLACE operations

Action Semantics
================
The MultiDiscrete action space has 5 dimensions:

  ======== =========== ======================== ============================
  Index    Name        Values                   Semantic
  ======== =========== ======================== ============================
  0        Operation   0=Place, 1=Store, 2=Ret  Which buffer/pallet op
  1        Slot        0..9                     Buffer slot index
  2        Grid X      0..15  (16 cells)        Pallet X position
  3        Grid Y      0..23  (24 cells)        Pallet Y position
  4        Rotation    0=0°, 1=90°              Z-axis rotation
  ======== =========== ======================== ============================

Grid → World Mapping
====================
  - Pallet size = (1.2, 0.8) m, centered at origin.
  - grid_x: 0..15 → X: -0.6..+0.6 m  (step = 0.075 m)
  - grid_y: 0..23 → Y: -0.4..+0.4 m  (step = 0.0333 m)
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pallet_rl.envs.pallet_task import PalletTask


# =============================================================================
# Public API
# =============================================================================

def pre_physics_step(env: PalletTask, actions: torch.Tensor) -> None:
    """
    Normalize/convert actions before physics stepping.

    Handles two formats (backwards-compatible):
      1. Discrete indices from MultiCategorical policy → center-of-bin normalize.
      2. Already-normalized float actions in [-1, 1] → clamp.
    """
    actions = actions.to(env._device).float()

    is_discrete = actions.abs().max() > 1.5

    if is_discrete:
        dims = env.cfg.action_dims
        for col, k in enumerate(dims):
            actions[:, col] = (actions[:, col].float() + 0.5) / k * 2.0 - 1.0

    env._actions = torch.clamp(actions, -1.0, 1.0)


def apply_action(env: PalletTask) -> None:
    """
    Decode continuous action and apply placement to simulation.

    Steps:
      1. Decode continuous [-1, 1] → discrete sub-actions.
      2. Compute target world position from action grid.
      3. Validate height constraint.
      4. Write pose to simulation for valid PLACE actions.
      5. Update payload mass and arm settling window.
      6. Delegate buffer operations to ``buffer_logic``.
    """
    from pallet_rl.envs.buffer_logic import handle_buffer_actions

    n = env.num_envs
    device = env._device
    cfg = env.cfg
    action = env._actions

    # ------------------------------------------------------------------
    # 1. Decode sub-actions
    # ------------------------------------------------------------------
    op_type = _to_discrete(action[:, 0], cfg.action_dims[0])
    slot_idx = _to_discrete(action[:, 1], cfg.action_dims[1])
    rot_idx = _to_discrete(action[:, 4], cfg.action_dims[4])
    grid_x = _to_discrete(action[:, 2], cfg.action_dims[2])
    grid_y = _to_discrete(action[:, 3], cfg.action_dims[3])

    # ------------------------------------------------------------------
    # 2. Grid → world coordinates
    # ------------------------------------------------------------------
    pallet_x, pallet_y = cfg.pallet_size
    num_x, num_y = cfg.action_dims[2], cfg.action_dims[3]
    step_x = pallet_x / num_x
    step_y = pallet_y / num_y
    half_x = pallet_x / 2.0
    half_y = pallet_y / 2.0

    target_x = grid_x.float() * step_x - half_x + step_x / 2
    target_y = grid_y.float() * step_y - half_y + step_y / 2
    target_z = torch.full((n,), 1.5, device=device)

    # ------------------------------------------------------------------
    # 3. Operation masks
    # ------------------------------------------------------------------
    env.active_place_mask = (op_type == 0) | (op_type == 2)
    env.store_mask = op_type == 1
    env.retrieve_mask = op_type == 2

    # ------------------------------------------------------------------
    # 4. Height constraint validation
    # ------------------------------------------------------------------
    _validate_height_constraint(
        env, op_type, slot_idx, grid_x, grid_y, n, device,
    )

    place_only_mask = (op_type == 0) & ~env._height_invalid_mask

    # ------------------------------------------------------------------
    # 5. Build target pose and rotation quaternion
    # ------------------------------------------------------------------
    target_pos = torch.stack([target_x, target_y, target_z], dim=-1)

    quat = torch.zeros(n, 4, device=device)
    quat[:, 0] = 1.0
    rot_mask = rot_idx == 1
    quat[rot_mask, 0] = 0.7071068
    quat[rot_mask, 3] = 0.7071068

    # Save for stability checks
    env.last_target_pos[:, 0] = target_x
    env.last_target_pos[:, 1] = target_y
    env.last_target_pos[:, 2] = 0.0
    env.last_target_quat.copy_(quat)

    # ------------------------------------------------------------------
    # 6. Write poses for PLACE actions
    # ------------------------------------------------------------------
    if "boxes" in env.scene.keys() and place_only_mask.any():
        _write_place_poses(env, place_only_mask, target_pos, quat, n, device)

    # ------------------------------------------------------------------
    # 7. Buffer logic (store / retrieve)
    # ------------------------------------------------------------------
    handle_buffer_actions(env, action)


def get_action_mask(env: PalletTask) -> torch.Tensor:
    """
    Return an action mask over the flattened MultiDiscrete logits.

    Shape: ``(num_envs, sum(action_dims))``, dtype=bool.

    Implements:
      - Height-based X/Y masking for PLACE operations.
      - Informational (not enforced) for RETRIEVE due to MultiDiscrete coupling.
    """
    n = env.num_envs
    device = env._device
    cfg = env.cfg
    total_logits = sum(cfg.action_dims)

    mask = torch.ones(n, total_logits, dtype=torch.bool, device=device)

    slot_start = cfg.action_dims[0]
    x_start = slot_start + cfg.action_dims[1]
    y_start = x_start + cfg.action_dims[2]

    if env._last_heightmap is not None:
        num_x = cfg.action_dims[2]
        num_y = cfg.action_dims[3]

        grid_xs = torch.arange(num_x, device=device)
        grid_ys = torch.arange(num_y, device=device)

        pixel_xs = (grid_xs.float() / max(1, num_x - 1) * (cfg.map_shape[1] - 1)).long()
        pixel_ys = (grid_ys.float() / max(1, num_y - 1) * (cfg.map_shape[0] - 1)).long()

        pixel_xs = pixel_xs.clamp(0, cfg.map_shape[1] - 1)
        pixel_ys = pixel_ys.clamp(0, cfg.map_shape[0] - 1)

        idx = env.box_idx.clamp(0, cfg.max_boxes - 1)
        env_idx = torch.arange(n, device=device)
        fresh_box_h = env.box_dims[env_idx, idx, 2]

        heightmap = env._last_heightmap
        all_heights = heightmap[:, pixel_ys[:, None], pixel_xs[None, :]]
        predicted_tops = all_heights + fresh_box_h[:, None, None]

        grid_invalid = predicted_tops > cfg.max_stack_height

        all_y_invalid_at_x = grid_invalid.all(dim=1)
        mask[:, x_start:x_start + num_x] &= ~all_y_invalid_at_x

        all_x_invalid_at_y = grid_invalid.all(dim=2)
        mask[:, y_start:y_start + num_y] &= ~all_x_invalid_at_y

    assert mask.device == device, "Action mask device mismatch"
    return mask


# =============================================================================
# Legacy decode_action (moved from algo/utils.py, kept for reference)
# =============================================================================

def decode_action(
    action_idx: torch.Tensor | int,
    width: int,
    height: int,
    num_rotations: int,
):
    """
    Decode a flat action index into ``(rot_idx, x, y)`` coordinates.

    Formula: ``index = rot * (H * W) + y * W + x``

    This is the legacy single-index spatial decoder, kept for backward
    compatibility and test reference.  The canonical pipeline uses the
    MultiDiscrete 5-tuple approach.
    """
    assert width > 0 and height > 0, f"Invalid grid dimensions: {width}x{height}"
    area = width * height
    rot_idx = action_idx // area
    spatial_idx = action_idx % area
    y = spatial_idx // width
    x = spatial_idx % width
    return rot_idx, x, y


# =============================================================================
# Internal Helpers
# =============================================================================

def _to_discrete(a: torch.Tensor, k: int) -> torch.Tensor:
    """Map [-1, 1] → {0 .. k-1}."""
    return torch.floor(((a + 1.0) * 0.5) * k).long().clamp(0, k - 1)


def _validate_height_constraint(
    env: PalletTask,
    op_type: torch.Tensor,
    slot_idx: torch.Tensor,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    n: int,
    device,
):
    """Validate height constraint for PLACE and RETRIEVE actions."""
    cfg = env.cfg
    env._height_invalid_mask[:] = False

    place_mask = op_type == 0
    retrieve_mask_local = op_type == 2
    needs_height_check = place_mask | retrieve_mask_local

    if env._last_heightmap is None or not needs_height_check.any():
        return

    pixel_x = (grid_x.float() / max(1, cfg.action_dims[2] - 1) * (cfg.map_shape[1] - 1)).long()
    pixel_y = (grid_y.float() / max(1, cfg.action_dims[3] - 1) * (cfg.map_shape[0] - 1)).long()
    pixel_x = pixel_x.clamp(0, cfg.map_shape[1] - 1)
    pixel_y = pixel_y.clamp(0, cfg.map_shape[0] - 1)

    env_idx = torch.arange(n, device=device)
    local_height = env._last_heightmap[env_idx, pixel_y, pixel_x]

    # PLACE: fresh box height
    box_idx_clamped = env.box_idx.clamp(0, cfg.max_boxes - 1)
    place_box_height = env.box_dims[env_idx, box_idx_clamped, 2]

    # RETRIEVE: buffered box height
    slot_idx_clamped = slot_idx.clamp(0, cfg.buffer_slots - 1)
    retrieve_box_height = env.buffer_state[env_idx, slot_idx_clamped, 2]

    box_height = torch.where(place_mask, place_box_height, retrieve_box_height)
    predicted_top = local_height + box_height

    height_exceeds = predicted_top > cfg.max_stack_height
    env._height_invalid_mask = height_exceeds & needs_height_check


def _write_place_poses(
    env: PalletTask,
    place_only_mask: torch.Tensor,
    target_pos: torch.Tensor,
    quat: torch.Tensor,
    n: int,
    device,
):
    """Write box poses to simulation for valid PLACE actions."""
    cfg = env.cfg
    place_envs = place_only_mask.nonzero(as_tuple=False).flatten()
    box_ids = env.box_idx[place_envs]

    flat_idx = (place_envs * cfg.max_boxes + box_ids).long()

    boxes_data = env.scene["boxes"].data
    pos = boxes_data.object_pos_w
    quat_w = boxes_data.object_quat_w
    lin = boxes_data.object_lin_vel_w
    ang = boxes_data.object_ang_vel_w

    B = pos.shape[1]
    env_ids = flat_idx // B
    box_ids2 = flat_idx % B

    pos[env_ids, box_ids2, :] = target_pos[place_envs]
    quat_w[env_ids, box_ids2, :] = quat[place_envs]
    lin[env_ids, box_ids2, :] = 0.0
    ang[env_ids, box_ids2, :] = 0.0

    env.scene["boxes"].write_data_to_sim()

    # Payload update
    placed_box_mass = env.box_mass_kg[place_envs, env.box_idx[place_envs]]
    env.payload_kg[place_envs] += placed_box_mass

    # Arm settling window
    env._settle_countdown[place_envs] = cfg.settle_steps
    env._settle_box_id[place_envs] = env.box_idx[place_envs]
    env._settle_target_pos[place_envs] = env.last_target_pos[place_envs]
    env._settle_target_quat[place_envs] = quat[place_envs]
