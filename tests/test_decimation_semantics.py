"""
Regression tests for DirectRLEnv decimation semantics.

These tests do NOT require Isaac Lab. They exercise the placement
controller and buffer logic using a lightweight mock environment to
prove that one RL action cannot increment ``box_idx`` or mutate payload
multiple times within a single RL step, even if ``_apply_action`` were
to be called repeatedly by the DirectRLEnv lifecycle.
"""

from __future__ import annotations

import torch

from pallet_rl.envs.placement_controller import pre_physics_step
from pallet_rl.envs.buffer_logic import handle_buffer_actions


class MockCfg:
    def __init__(self):
        self.action_dims = (3, 10, 16, 24, 2)
        self.pallet_size = (1.2, 0.8)
        self.max_stack_height = 1.8
        self.map_shape = (160, 240)
        self.buffer_slots = 10
        self.max_boxes = 50
        self.num_boxes = 10  # small episode to stress num_vs_max semantics


class MockEnv:
    """Minimal mock that exposes the attributes used by the controllers."""

    def __init__(self, num_envs: int = 2, device: str = "cpu"):
        self.num_envs = num_envs
        self._device = torch.device(device)
        self.cfg = MockCfg()

        n = num_envs
        M = self.cfg.max_boxes
        self._actions = torch.zeros(n, 5, device=self._device)
        self.decoded_action = None

        self.box_idx = torch.zeros(n, dtype=torch.long, device=self._device)
        self.box_dims = torch.ones(n, M, 3, device=self._device) * 0.2
        self.box_mass_kg = torch.ones(n, M, device=self._device) * 5.0
        self.payload_kg = torch.zeros(n, device=self._device)

        self.buffer_state = torch.zeros(n, self.cfg.buffer_slots, 6, device=self._device)
        self.buffer_has_box = torch.zeros(n, self.cfg.buffer_slots, dtype=torch.bool, device=self._device)
        self.buffer_box_id = torch.full((n, self.cfg.buffer_slots), -1, dtype=torch.long, device=self._device)

        self.active_place_mask = torch.zeros(n, dtype=torch.bool, device=self._device)
        self.store_mask = torch.zeros(n, dtype=torch.bool, device=self._device)
        self.retrieve_mask = torch.zeros(n, dtype=torch.bool, device=self._device)
        self.valid_retrieve = torch.zeros(n, dtype=torch.bool, device=self._device)
        self.valid_store = torch.zeros(n, dtype=torch.bool, device=self._device)
        self._height_invalid_mask = torch.zeros(n, dtype=torch.bool, device=self._device)

        self.last_moved_box_id = torch.full((n,), -1, dtype=torch.long, device=self._device)
        self.last_target_pos = torch.zeros(n, 3, device=self._device)
        self.last_target_quat = torch.zeros(n, 4, device=self._device)
        self.last_target_quat[:, 0] = 1.0

        # Heightmap: flat floor at z=0 so all placements are valid.
        H, W = self.cfg.map_shape
        self._last_heightmap = torch.zeros(n, H, W, device=self._device)

        # Settling bookkeeping (not used by this test but required fields)
        self._settle_countdown = torch.zeros(n, dtype=torch.long, device=self._device)
        self._settle_box_id = torch.full((n,), -1, dtype=torch.long, device=self._device)
        self._settle_target_pos = torch.zeros(n, 3, device=self._device)
        self._settle_target_quat = torch.zeros(n, 4, device=self._device)

        # Scene stub for placement controller; we never actually write poses.
        class _BoxesData:
            def __init__(self, n_envs: int, max_boxes: int, dev):
                self.object_pos_w = torch.zeros(n_envs, max_boxes, 3, device=dev)
                self.object_quat_w = torch.zeros(n_envs, max_boxes, 4, device=dev)
                self.object_lin_vel_w = torch.zeros(n_envs, max_boxes, 3, device=dev)
                self.object_ang_vel_w = torch.zeros(n_envs, max_boxes, 3, device=dev)

        class _Boxes:
            def __init__(self, n_envs: int, max_boxes: int, dev):
                self.data = _BoxesData(n_envs, max_boxes, dev)

            def write_data_to_sim(self):
                # No-op in tests
                pass

        self.scene = {"boxes": _Boxes(n, M, self._device)}


def test_single_rl_step_does_not_double_apply_side_effects():
    """
    Given one RL action, calling the placement+buffer logic multiple
    times must NOT increment box_idx or payload more than once.
    """
    env = MockEnv(num_envs=2)

    # One PLACE action per env: op=0 (place), slot arbitrary, x,y center, rot=0
    discrete_actions = torch.tensor([[0, 0, 8, 12, 0], [0, 1, 5, 10, 1]], dtype=torch.float32)

    # Commit RL action once per step
    pre_physics_step(env, discrete_actions)

    # Snapshot state after the RL-step commit
    box_idx_before = env.box_idx.clone()
    payload_before = env.payload_kg.clone()

    # Simulate DirectRLEnv repeatedly calling buffer logic (old bug would
    # have advanced box_idx and payload multiple times).
    for _ in range(5):
        handle_buffer_actions(env)

    # Each env must have advanced by exactly one PLACE and payload once.
    assert torch.all(env.box_idx - box_idx_before == 1), "box_idx must increment exactly once per RL action"
    expected_payload = payload_before + env.box_mass_kg[:, 0]
    assert torch.allclose(env.payload_kg, expected_payload), "payload must update exactly once per RL action"


def test_box_idx_respects_num_boxes():
    """
    When num_boxes < max_boxes, repeated PLACE actions must not advance
    box_idx beyond num_boxes, and inactive boxes must remain inactive.
    """
    env = MockEnv(num_envs=1)
    env.cfg.num_boxes = 3

    # Apply more PLACE actions than num_boxes allows.
    discrete_actions = torch.tensor([[0, 0, 8, 12, 0]], dtype=torch.float32, device=env._device)
    for _ in range(10):
        pre_physics_step(env, discrete_actions)
        handle_buffer_actions(env)

    assert int(env.box_idx.item()) == env.cfg.num_boxes, "box_idx must be clamped at num_boxes"

