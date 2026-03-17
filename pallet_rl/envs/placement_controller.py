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
The factored discrete action space has 5 dimensions:

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

This component is the AUTHORITATIVE source of truth for action validity 
and environment-side physical constraints.
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
    RL-step action commit for DirectRLEnv.

    DirectRLEnv calls ``_pre_physics_step`` once per RL step but may call
    ``_apply_action`` on every physics substep when ``decimation > 1``.
    To avoid repeated side effects within a single RL step, all placement
    and buffer bookkeeping is performed here, and ``_apply_action`` is
    intentionally left as a no-op in :mod:`pallet_task`.

    Handles two input formats (backwards-compatible):
      1. Discrete indices from MultiCategorical policy → center-of-bin normalize.
      2. Already-normalized float actions in [-1, 1] → clamp.
    """
    from pallet_rl.envs.buffer_logic import handle_buffer_actions
    from pallet_rl.envs.action_adapter import (
        decode_action_tensor,
        is_discrete_action_tensor,
        discrete_to_center_normalized
    )

    # ------------------------------------------------------------------
    # 0. Normalize/convert actions once per RL step
    # ------------------------------------------------------------------
    actions = actions.to(env._device)
    dims = env.cfg.action_dims

    # Centralized decoding ensures [[1,1,1,1,1]] is correctly treated as discrete
    env.decoded_action = decode_action_tensor(actions, dims)

    # Normalize actions internal state for consistency/logging if needed
    if is_discrete_action_tensor(actions, dims):
        env._actions = discrete_to_center_normalized(actions, dims)
    else:
        env._actions = torch.clamp(actions.float(), -1.0, 1.0)

    # ------------------------------------------------------------------
    # 1. Decode sub-actions explicitly (factored discrete)
    # ------------------------------------------------------------------
    n = env.num_envs
    device = env._device
    cfg = env.cfg
    
    dec = env.decoded_action
    op_type = dec.op_type
    slot_idx = dec.slot_idx
    rot_idx = dec.rot_idx
    grid_x = dec.grid_x
    grid_y = dec.grid_y

    # ------------------------------------------------------------------
    # 2. Operation masks (per RL step)
    # ------------------------------------------------------------------
    env.active_place_mask = (op_type == 0)
    env.active_motion_mask = (op_type == 0) | (op_type == 2)
    env.store_mask = op_type == 1
    env.retrieve_mask = op_type == 2

    # ------------------------------------------------------------------
    # 3. Height constraint validation (PLACE + RETRIEVE)
    # ------------------------------------------------------------------
    _validate_height_constraint(
        env, op_type, slot_idx, grid_x, grid_y, n, device,
    )

    place_only_mask = (op_type == 0) & ~env._height_invalid_mask

    # ------------------------------------------------------------------
    # 4. Grid → world coordinates
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
    # 6. Write poses for PLACE actions (one-shot)
    # ------------------------------------------------------------------
    if "boxes" in env.scene.keys() and place_only_mask.any():
        _write_place_poses(env, place_only_mask, target_pos, quat, n, device)

    # ------------------------------------------------------------------
    # 7. Buffer logic (store / retrieve, box_idx updates)
    # ------------------------------------------------------------------
    handle_buffer_actions(env)


def get_action_mask(env: PalletTask) -> torch.Tensor:
    """
    Return an action mask over the flattened discrete logits.

    Shape: ``(num_envs, sum(action_dims))``, dtype=bool.

    Implements:
      - Height-based X/Y masking for PLACE operations.
      - Informational (not enforced) for RETRIEVE due to independent component sampling coupling.
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
    factored discrete 5-tuple approach.
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

    # Map discrete grid indices to heightmap pixel coordinates
    pixel_x = (grid_x.float() / max(1, cfg.action_dims[2] - 1) * (cfg.map_shape[1] - 1)).long()
    pixel_y = (grid_y.float() / max(1, cfg.action_dims[3] - 1) * (cfg.map_shape[0] - 1)).long()
    pixel_x = pixel_x.clamp(0, cfg.map_shape[1] - 1)
    pixel_y = pixel_y.clamp(0, cfg.map_shape[0] - 1)

    env_idx = torch.arange(n, device=device)

    # PLACE: fresh box height and footprint
    box_idx_clamped = env.box_idx.clamp(0, cfg.max_boxes - 1)
    place_box_dims = env.box_dims[env_idx, box_idx_clamped]  # (N, 3)
    place_box_height = place_box_dims[:, 2]

    # RETRIEVE: buffered box height and footprint
    slot_idx_clamped = slot_idx.clamp(0, cfg.buffer_slots - 1)
    retrieve_box_dims = env.buffer_state[env_idx, slot_idx_clamped, :3]
    retrieve_box_height = retrieve_box_dims[:, 2]

    # SELECT footprint dims per-op (L,W,H)
    box_dims = torch.where(
        place_mask.unsqueeze(-1),
        place_box_dims,
        retrieve_box_dims,
    )
    box_height = torch.where(place_mask, place_box_height, retrieve_box_height)

    # Approximate footprint-based validation by sampling a 3×3 stencil
    # around the target pixel, using the current box height. This is
    # substantially more robust than a single-point check while keeping
    # the computation lightweight and GPU-friendly.
    H, W = cfg.map_shape
    offsets = torch.tensor([-1, 0, 1], device=device, dtype=torch.long)
    # Cartesian product for a true 3x3 neighborhood
    grid_off_y, grid_off_x = torch.meshgrid(offsets, offsets, indexing="ij")
    grid_off_y = grid_off_y.reshape(-1)
    grid_off_x = grid_off_x.reshape(-1)

    off_y = (pixel_y[:, None] + grid_off_y[None, :]).clamp(0, H - 1)
    off_x = (pixel_x[:, None] + grid_off_x[None, :]).clamp(0, W - 1)

    # Gather local 3×3 (9 points) neighborhood heights
    hmap = env._last_heightmap
    local_heights = hmap[env_idx[:, None], off_y, off_x]  # (N, 9)
    max_local_height = local_heights.max(dim=1).values

    predicted_top = max_local_height + box_height

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
