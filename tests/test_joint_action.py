"""
Unit tests for the joint place-only action space.
Does NOT require Isaac Lab.
"""

import torch
import pytest
from pallet_rl.envs.action_adapter import (
    decode_joint_place_action,
    encode_joint_place_action,
    DecodedPlaceAction,
)


# =========================================================================
# Round-trip encoding/decoding
# =========================================================================

class TestJointActionRoundTrip:
    """Verify encode → decode and decode → encode are inverses."""

    def test_single_action_roundtrip(self):
        gx, gy, nr = 8, 12, 2
        for idx in [0, 1, 95, 96, 191]:
            action_idx = torch.tensor([idx])
            dec = decode_joint_place_action(action_idx, gx, gy, nr)
            re_encoded = encode_joint_place_action(
                dec.grid_x, dec.grid_y, dec.rot_idx, gx, gy
            )
            assert re_encoded.item() == idx, f"Round-trip failed for idx={idx}"

    def test_batch_roundtrip(self):
        gx, gy, nr = 8, 12, 2
        total = gx * gy * nr
        action_idx = torch.arange(total)
        dec = decode_joint_place_action(action_idx, gx, gy, nr)
        re_encoded = encode_joint_place_action(
            dec.grid_x, dec.grid_y, dec.rot_idx, gx, gy
        )
        assert torch.equal(re_encoded, action_idx)

    def test_all_stage_grids(self):
        """Test round-trip for all curriculum stage grid sizes."""
        grids = [(8, 12, 2), (12, 18, 2), (16, 24, 2)]
        for gx, gy, nr in grids:
            total = gx * gy * nr
            action_idx = torch.arange(total)
            dec = decode_joint_place_action(action_idx, gx, gy, nr)
            re_encoded = encode_joint_place_action(
                dec.grid_x, dec.grid_y, dec.rot_idx, gx, gy
            )
            assert torch.equal(re_encoded, action_idx), f"Failed for grid ({gx},{gy},{nr})"


# =========================================================================
# Decoding correctness
# =========================================================================

class TestDecodeJointPlaceAction:
    def test_first_action(self):
        """Index 0 should map to (x=0, y=0, rot=0)."""
        dec = decode_joint_place_action(torch.tensor([0]), 8, 12, 2)
        assert dec.grid_x.item() == 0
        assert dec.grid_y.item() == 0
        assert dec.rot_idx.item() == 0

    def test_last_action_rot0(self):
        """Last action in rot=0 block = index (8*12 - 1) = 95."""
        gx, gy = 8, 12
        dec = decode_joint_place_action(torch.tensor([gx * gy - 1]), gx, gy, 2)
        assert dec.grid_x.item() == gx - 1  # 7
        assert dec.grid_y.item() == gy - 1  # 11
        assert dec.rot_idx.item() == 0

    def test_first_action_rot1(self):
        """First action in rot=1 block = index (8*12) = 96."""
        gx, gy = 8, 12
        dec = decode_joint_place_action(torch.tensor([gx * gy]), gx, gy, 2)
        assert dec.grid_x.item() == 0
        assert dec.grid_y.item() == 0
        assert dec.rot_idx.item() == 1

    def test_last_action(self):
        """Last action overall = (8*12*2 - 1) = 191."""
        gx, gy = 8, 12
        dec = decode_joint_place_action(torch.tensor([gx * gy * 2 - 1]), gx, gy, 2)
        assert dec.grid_x.item() == gx - 1
        assert dec.grid_y.item() == gy - 1
        assert dec.rot_idx.item() == 1

    def test_encoding_order(self):
        """Verify encoding is: index = rot * (gy * gx) + y * gx + x."""
        gx, gy = 8, 12
        x, y, r = 3, 5, 1
        expected = r * (gy * gx) + y * gx + x
        dec = decode_joint_place_action(torch.tensor([expected]), gx, gy, 2)
        assert dec.grid_x.item() == x
        assert dec.grid_y.item() == y
        assert dec.rot_idx.item() == r

    def test_batch_sizes(self):
        """Verify batch dimension is preserved."""
        action_idx = torch.tensor([0, 50, 100, 191])
        dec = decode_joint_place_action(action_idx, 8, 12, 2)
        assert dec.grid_x.shape == (4,)
        assert dec.grid_y.shape == (4,)
        assert dec.rot_idx.shape == (4,)

    def test_clamping(self):
        """Out-of-range indices should be clamped."""
        gx, gy, nr = 8, 12, 2
        total = gx * gy * nr
        dec = decode_joint_place_action(torch.tensor([total + 10]), gx, gy, nr)
        assert dec.rot_idx.item() <= nr - 1
        assert dec.grid_x.item() <= gx - 1
        assert dec.grid_y.item() <= gy - 1


# =========================================================================
# Encoding correctness
# =========================================================================

class TestEncodeJointPlaceAction:
    def test_origin(self):
        idx = encode_joint_place_action(
            torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), 8, 12
        )
        assert idx.item() == 0

    def test_specific_cell(self):
        """x=3, y=5, rot=1 in 8x12 grid → 1*96 + 5*8 + 3 = 139."""
        idx = encode_joint_place_action(
            torch.tensor([3]), torch.tensor([5]), torch.tensor([1]), 8, 12
        )
        assert idx.item() == 1 * 96 + 5 * 8 + 3  # 139

    def test_batch(self):
        gx = torch.tensor([0, 7, 3])
        gy = torch.tensor([0, 11, 5])
        rot = torch.tensor([0, 0, 1])
        idx = encode_joint_place_action(gx, gy, rot, 8, 12)
        assert idx.shape == (3,)
        assert idx[0].item() == 0
        assert idx[1].item() == 11 * 8 + 7  # 95
        assert idx[2].item() == 96 + 5 * 8 + 3  # 139


# =========================================================================
# Action mask shape
# =========================================================================

class TestActionMaskShape:
    """Test that action mask helpers produce correct shapes."""

    def test_place_only_mask_basic(self):
        """Verify get_action_mask_place_only produces correct shape."""
        from pallet_rl.envs.placement_controller import get_action_mask_place_only

        class MockCfg:
            place_only_grid = (8, 12)
            place_only_rotations = 2
            pallet_size = (1.2, 0.8)
            place_border_epsilon_m = 1e-6
            max_boxes = 50
            max_stack_height = 0.6
            map_shape = (160, 240)

        class MockEnv:
            num_envs = 2
            _device = torch.device("cpu")
            cfg = MockCfg()
            box_idx = torch.zeros(2, dtype=torch.long)
            box_dims = torch.zeros(2, 50, 3)
            _last_heightmap = None

        env = MockEnv()
        env.box_dims[:, 0] = torch.tensor([0.4, 0.3, 0.2])

        mask = get_action_mask_place_only(env)
        assert mask.shape == (2, 8 * 12 * 2)
        assert mask.dtype == torch.bool
        # At least some actions should be valid
        assert mask.any()

    def test_border_masking_large_box(self):
        """A very large box should mask most edge cells."""
        from pallet_rl.envs.placement_controller import get_action_mask_place_only

        class MockCfg:
            place_only_grid = (8, 12)
            place_only_rotations = 2
            pallet_size = (1.2, 0.8)
            place_border_epsilon_m = 1e-6
            max_boxes = 50
            max_stack_height = 2.0
            map_shape = (160, 240)

        class MockEnv:
            num_envs = 1
            _device = torch.device("cpu")
            cfg = MockCfg()
            box_idx = torch.zeros(1, dtype=torch.long)
            box_dims = torch.zeros(1, 50, 3)
            _last_heightmap = None

        env = MockEnv()
        # Very large box: only center cells viable
        env.box_dims[0, 0] = torch.tensor([1.0, 0.6, 0.2])

        mask = get_action_mask_place_only(env)
        total = 8 * 12 * 2
        # Large box should have fewer valid actions than small box
        valid_count = mask.sum().item()
        assert valid_count < total
        assert valid_count > 0  # At least center should work


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
