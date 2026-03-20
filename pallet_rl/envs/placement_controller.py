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

    num_x = cfg.action_dims[2]
    num_y = cfg.action_dims[3]

    # --- 1. Conservative Border Mask for PLACE ---
    # We use current fresh box dims and check if X/Y centers are valid for EITHER rotation.
    grid_xs = torch.arange(num_x, device=device)
    grid_ys = torch.arange(num_y, device=device)
    
    # Centers
    cxs, cys = _grid_to_pallet_center(grid_xs, grid_ys, cfg.pallet_size, cfg.action_dims)
    
    # Box dims for current fresh box
    idx = env.box_idx.clamp(0, cfg.max_boxes - 1)
    env_idx = torch.arange(n, device=device)
    fresh_box_dims = env.box_dims[env_idx, idx] # (N, 3)
    
    # Effective XY for both rotations
    eff_xy_rot0 = _get_effective_xy_dims(fresh_box_dims, 0) # (N, 2)
    eff_xy_rot1 = _get_effective_xy_dims(fresh_box_dims, 1) # (N, 2)
    
    px, py = cfg.pallet_size
    eps = cfg.place_border_epsilon_m

    # X validity: (N, num_x)
    # Valid if center is okay for EITHER rot0 OR rot1
    dx0, dx1 = eff_xy_rot0[:, 0:1], eff_xy_rot1[:, 0:1] # (N, 1)
    x_valid_rot0 = (cxs[None, :] - dx0/2 >= -px/2 + eps) & (cxs[None, :] + dx0/2 <= px/2 - eps)
    x_valid_rot1 = (cxs[None, :] - dx1/2 >= -px/2 + eps) & (cxs[None, :] + dx1/2 <= px/2 - eps)
    x_border_valid = x_valid_rot0 | x_valid_rot1
    mask[:, x_start : x_start + num_x] &= x_border_valid

    # Y validity: (N, num_y)
    dy0, dy1 = eff_xy_rot0[:, 1:2], eff_xy_rot1[:, 1:2] # (N, 1)
    y_valid_rot0 = (cys[None, :] - dy0/2 >= -py/2 + eps) & (cys[None, :] + dy0/2 <= py/2 - eps)
    y_valid_rot1 = (cys[None, :] - dy1/2 >= -py/2 + eps) & (cys[None, :] + dy1/2 <= py/2 - eps)
    y_border_valid = y_valid_rot0 | y_valid_rot1
    mask[:, y_start : y_start + num_y] &= y_border_valid

    # --- 2. Height-based X/Y masking for PLACE ---
    if env._last_heightmap is not None:
        pixel_xs = (grid_xs.float() / max(1, num_x - 1) * (cfg.map_shape[1] - 1)).long()
        pixel_ys = (grid_ys.float() / max(1, num_y - 1) * (cfg.map_shape[0] - 1)).long()

        pixel_xs = pixel_xs.clamp(0, cfg.map_shape[1] - 1)
        pixel_ys = pixel_ys.clamp(0, cfg.map_shape[0] - 1)

        fresh_box_h = fresh_box_dims[:, 2]

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

def _grid_to_pallet_center(
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    pallet_size: tuple[float, float],
    action_dims: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert grid indices to pallet-centered coordinates."""
    pallet_x, pallet_y = pallet_size
    num_x, num_y = action_dims[2], action_dims[3]
    step_x = pallet_x / num_x
    step_y = pallet_y / num_y
    half_x = pallet_x / 2.0
    half_y = pallet_y / 2.0

    center_x = grid_x.float() * step_x - half_x + step_x / 2
    center_y = grid_y.float() * step_y - half_y + step_y / 2
    return center_x, center_y


def _get_effective_xy_dims(
    box_dims: torch.Tensor,
    rot_idx: torch.Tensor,
) -> torch.Tensor:
    """Compute effective XY footprint dimensions from box dims and rot_idx."""
    # box_dims: (N, 3) or (3,)
    # rot_idx: (N,) or int
    
    # If rot_idx == 0: (box_dim_x, box_dim_y)
    # If rot_idx == 1: swap x and y
    
    # Ensure tensor shape handles both single env and batch
    if box_dims.dim() == 1:
        box_dims = box_dims.unsqueeze(0)
    if not isinstance(rot_idx, torch.Tensor):
        rot_idx = torch.tensor([rot_idx], device=box_dims.device)
    
    effective_xy = box_dims[:, :2].clone()
    rot_mask = rot_idx == 1
    if rot_mask.any():
        effective_xy[rot_mask, 0] = box_dims[rot_mask, 1]
        effective_xy[rot_mask, 1] = box_dims[rot_mask, 0]
    
    return effective_xy


def _check_inside_pallet(
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    effective_xy: torch.Tensor,
    pallet_size: tuple[float, float],
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Check whether a rotated footprint is fully inside the pallet."""
    px, py = pallet_size
    dx, dy = effective_xy[:, 0], effective_xy[:, 1]
    
    valid = (center_x - dx/2 >= -px/2 + epsilon) & \
            (center_x + dx/2 <=  px/2 - epsilon) & \
            (center_y - dy/2 >= -py/2 + epsilon) & \
            (center_y + dy/2 <=  py/2 - epsilon)
    return valid


def _get_footprint_pixel_bounds(
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    effective_xy: torch.Tensor,
    pallet_size: tuple[float, float],
    map_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map pallet XY footprint bounds to clamped heightmap pixel bounds (inclusive)."""
    px, py = pallet_size
    H, W = map_shape
    dx, dy = effective_xy[:, 0], effective_xy[:, 1]
    
    # Pallet-centered to normalized [0, 1]
    # x: [-px/2, px/2] -> [0, 1] => (x + px/2) / px
    x_min = (center_x - dx/2 + px/2) / px
    x_max = (center_x + dx/2 + px/2) / px
    y_min = (center_y - dy/2 + py/2) / py
    y_max = (center_y + dy/2 + py/2) / py
    
    # Normalized to pixel indices
    # Be conservative: floor for min, ceil for max
    ix_min = torch.floor(x_min * W).long()
    ix_max = torch.ceil(x_max * W).long()
    iy_min = torch.floor(y_min * H).long()
    iy_max = torch.ceil(y_max * H).long()
    
    # Clamp to valid image bounds
    ix_min = ix_min.clamp(0, W - 1)
    ix_max = ix_max.clamp(0, W - 1)
    iy_min = iy_min.clamp(0, H - 1)
    iy_max = iy_max.clamp(0, H - 1)
    
    return ix_min, ix_max, iy_min, iy_max


def _evaluate_patch_support(
    patch: torch.Tensor,
    box_height: float | torch.Tensor,
    max_stack_height: float,
    support_tol: float,
    support_ratio_min: float,
) -> torch.Tensor:
    """Evaluate support/height over one selected footprint patch."""
    if patch.numel() == 0:
        return torch.tensor(False, device=patch.device)
        
    local_top = patch.max()
    predicted_top = local_top + box_height
    
    support_mask = patch >= (local_top - support_tol)
    support_ratio = support_mask.float().mean()
    
    valid = (predicted_top <= max_stack_height) & (support_ratio >= support_ratio_min)
    return valid


def _validate_height_constraint(
    env: PalletTask,
    op_type: torch.Tensor,
    slot_idx: torch.Tensor,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    n: int,
    device,
):
    """
    Authoritative Selected-Action Feasibility Check for PLACE and RETRIEVE.
    
    Validates:
      1. Border check: Footprint must be fully inside pallet.
      2. Height check: Predicted top must not exceed max stack height.
      3. Support check: Sufficient fraction of footprint must be supported.
    """
    cfg = env.cfg
    env._height_invalid_mask[:] = False

    place_mask = op_type == 0
    retrieve_mask_local = op_type == 2
    needs_check = place_mask | retrieve_mask_local

    if env._last_heightmap is None or not needs_check.any():
        return

    check_indices = needs_check.nonzero(as_tuple=False).flatten()
    env_idx_tensor = torch.arange(n, device=device)

    # Dims per environment (PLACE or RETRIEVE)
    box_idx_clamped = env.box_idx.clamp(0, cfg.max_boxes - 1)
    slot_idx_clamped = slot_idx.clamp(0, cfg.buffer_slots - 1)
    
    place_box_dims = env.box_dims[env_idx_tensor, box_idx_clamped]
    retrieve_box_dims = env.buffer_state[env_idx_tensor, slot_idx_clamped, :3]
    
    env_box_dims = torch.where(
        place_mask.unsqueeze(-1),
        place_box_dims,
        retrieve_box_dims,
    )

    # Core loop over active checked envs (acceptable for runtime path as n_active is small)
    for i in check_indices:
        idx = i.item()
        
        # 1. Grid -> Pallet center
        cx, cy = _grid_to_pallet_center(
            grid_x[idx : idx + 1],
            grid_y[idx : idx + 1],
            cfg.pallet_size,
            cfg.action_dims,
        )
        
        # 2. Footprint dims
        eff_xy = _get_effective_xy_dims(
            env_box_dims[idx],
            env.decoded_action.rot_idx[idx],
        )
        
        # 3. Border check
        if not _check_inside_pallet(cx, cy, eff_xy, cfg.pallet_size, cfg.place_border_epsilon_m):
            env._height_invalid_mask[idx] = True
            continue
            
        # 4. Heightmap patch extraction
        ix_min, ix_max, iy_min, iy_max = _get_footprint_pixel_bounds(
            cx, cy, eff_xy, cfg.pallet_size, cfg.map_shape
        )
        
        # Slicing is inclusive
        patch = env._last_heightmap[idx, iy_min[0] : iy_max[0] + 1, ix_min[0] : ix_max[0] + 1]
        
        # 5. Support & Height evaluation
        valid = _evaluate_patch_support(
            patch,
            env_box_dims[idx, 2],
            cfg.max_stack_height,
            cfg.place_support_height_tol_m,
            cfg.place_support_ratio_min,
        )
        
        if not valid:
            env._height_invalid_mask[idx] = True


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
