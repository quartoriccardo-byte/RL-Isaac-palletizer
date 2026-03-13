"""
Buffer management logic for the PalletTask environment.

Handles store/retrieve operations on the buffer zone, including
physical box tracking, mass bookkeeping, and settling window arming.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pallet_rl.envs.pallet_task import PalletTask


def handle_buffer_actions(env: PalletTask):
    """
    Handle store and retrieve buffer operations with physical box tracking.

    Physical buffer semantics:
      - STORE parks an existing physical box in a holding area and records its ID.
      - RETRIEVE moves that same parked physical box back onto the pallet.
      - The buffer does NOT create new boxes.

    Also handles:
      - Mass tracking in ``buffer_state[:, :, 5]``.
      - Payload updates for STORE (remove from pallet) and RETRIEVE (add to pallet).
      - Settling window arm for RETRIEVE.
    """
    device = env._device
    cfg = env.cfg
    n = env.num_envs
    env_idx = torch.arange(n, device=device)

    dec = env.decoded_action
    slot_idx = dec.slot_idx
    op_type = dec.op_type

    # Reset last_moved_box_id; will be set appropriately below for
    # PLACE / RETRIEVE actions that actually move a physical box.
    env.last_moved_box_id[:] = -1

    # ==================================================================
    # STORE: Park the last-placed physical box in a buffer slot
    # ==================================================================
    has_box_to_store = env.box_idx > 0
    slot_is_empty = ~env.buffer_has_box[env_idx, slot_idx]
    valid_store = env.store_mask & has_box_to_store & slot_is_empty
    env.valid_store = valid_store

    if valid_store.any():
        _execute_store(env, valid_store, slot_idx, device)

    # ==================================================================
    # RETRIEVE: Move a parked physical box back onto the pallet
    # ==================================================================
    retrieve_height_valid = ~env._height_invalid_mask
    has_box_in_slot = env.buffer_has_box[env_idx, slot_idx]
    env.valid_retrieve = env.retrieve_mask & has_box_in_slot & retrieve_height_valid

    if env.valid_retrieve.any():
        _execute_retrieve(env, slot_idx, device)

    # Age all buffer slots
    env.buffer_state[:, :, 4] += 1.0

    # ==================================================================
    # PLACE: Advance box_idx and record last_moved_box_id
    # ==================================================================
    # Only PLACE operations with a remaining *fresh* box are allowed to
    # consume from the episode stream.  This enforces num_boxes vs
    # max_boxes semantics and prevents box_idx from advancing beyond the
    # active episode allocation.
    has_fresh_box = env.box_idx < cfg.num_boxes
    place_mask = env.active_place_mask & ~env._height_invalid_mask & has_fresh_box

    env.last_moved_box_id = torch.where(
        place_mask, env.box_idx, env.last_moved_box_id
    )

    # Clamp box_idx at num_boxes to keep inactive boxes truly inactive.
    env.box_idx = torch.minimum(
        env.box_idx + place_mask.long(),
        torch.full_like(env.box_idx, cfg.num_boxes),
    )


def _execute_store(
    env: PalletTask,
    valid_store: torch.Tensor,
    slot_idx: torch.Tensor,
    device,
):
    """Execute valid STORE operations: park boxes, update buffers, adjust payload."""
    cfg = env.cfg
    store_envs = valid_store.nonzero(as_tuple=False).flatten()
    store_slots = slot_idx[store_envs]

    stored_physical_id = (env.box_idx[store_envs] - 1).clamp(0, cfg.max_boxes - 1)
    dims = env.box_dims[store_envs, stored_physical_id]
    stored_mass = env.box_mass_kg[store_envs, stored_physical_id]

    # Update buffer_state
    env.buffer_state[store_envs, store_slots, :3] = dims
    env.buffer_state[store_envs, store_slots, 3] = 1.0
    env.buffer_state[store_envs, store_slots, 4] = 0.0
    env.buffer_state[store_envs, store_slots, 5] = stored_mass

    env.buffer_has_box[store_envs, store_slots] = True
    env.buffer_box_id[store_envs, store_slots] = stored_physical_id

    # Payload update: remove stored box mass
    env.payload_kg[store_envs] -= stored_mass

    # Move box off-map
    if "boxes" in env.scene.keys():
        global_store_idx = (store_envs * cfg.max_boxes + stored_physical_id).long()
        holding_pos = env._inactive_box_pos.expand(len(store_envs), 3)
        holding_quat = torch.zeros(len(store_envs), 4, device=device)
        holding_quat[:, 0] = 1.0

        boxes_data = env.scene["boxes"].data
        boxes_data.object_pos_w.reshape(-1, 3)[global_store_idx] = holding_pos
        boxes_data.object_quat_w.reshape(-1, 4)[global_store_idx] = holding_quat
        boxes_data.object_lin_vel_w.reshape(-1, 3)[global_store_idx] = 0.0
        boxes_data.object_ang_vel_w.reshape(-1, 3)[global_store_idx] = 0.0
        env.scene["boxes"].write_data_to_sim()


def _execute_retrieve(
    env: PalletTask,
    slot_idx: torch.Tensor,
    device,
):
    """Execute valid RETRIEVE operations: re-place buffered boxes, update payload."""
    cfg = env.cfg
    retr_envs = env.valid_retrieve.nonzero(as_tuple=False).flatten()
    retr_slots = slot_idx[retr_envs]

    retrieved_physical_id = env.buffer_box_id[retr_envs, retr_slots]
    retrieved_mass = env.buffer_state[retr_envs, retr_slots, 5]

    if "boxes" in env.scene.keys():
        global_retr_idx = retr_envs * cfg.max_boxes + retrieved_physical_id

        # Compute target pose from action grid
        pallet_x = cfg.pallet_size[0]
        pallet_y = cfg.pallet_size[1]
        num_x = cfg.action_dims[2]
        num_y = cfg.action_dims[3]

        step_x = pallet_x / num_x
        step_y = pallet_y / num_y
        half_x = pallet_x / 2.0
        half_y = pallet_y / 2.0

        dec = env.decoded_action
        grid_x = dec.grid_x[retr_envs]
        grid_y = dec.grid_y[retr_envs]
        rot_idx = dec.rot_idx[retr_envs]

        target_x = grid_x.float() * step_x - half_x + step_x / 2
        target_y = grid_y.float() * step_y - half_y + step_y / 2
        target_z = torch.full((len(retr_envs),), 1.5, device=device)
        target_pos = torch.stack([target_x, target_y, target_z], dim=-1)

        quat = torch.zeros(len(retr_envs), 4, device=device)
        quat[:, 0] = 1.0
        rot_mask = rot_idx == 1
        quat[rot_mask, 0] = 0.7071068
        quat[rot_mask, 3] = 0.7071068

        boxes_data = env.scene["boxes"].data
        boxes_data.object_pos_w.view(-1, 3)[global_retr_idx] = target_pos
        boxes_data.object_quat_w.view(-1, 4)[global_retr_idx] = quat
        boxes_data.object_lin_vel_w.view(-1, 3)[global_retr_idx] = 0.0
        boxes_data.object_ang_vel_w.view(-1, 3)[global_retr_idx] = 0.0
        env.scene["boxes"].write_data_to_sim()

        # Arm settling window
        env._settle_countdown[retr_envs] = cfg.settle_steps
        env._settle_box_id[retr_envs] = retrieved_physical_id
        settle_target = target_pos.clone()
        settle_target[:, 2] = 0.0
        env._settle_target_pos[retr_envs] = settle_target
        env._settle_target_quat[retr_envs] = quat

    # Payload update: add retrieved mass
    env.payload_kg[retr_envs] += retrieved_mass

    # Clear buffer slot
    env.buffer_has_box[retr_envs, retr_slots] = False
    env.buffer_box_id[retr_envs, retr_slots] = -1
    env.buffer_state[retr_envs, retr_slots] = 0.0

    env.last_moved_box_id[retr_envs] = retrieved_physical_id
    env.active_place_mask = env.active_place_mask | env.valid_retrieve
