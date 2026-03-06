"""
Unit tests for the extracted placement controller module.

These tests validate action decoding, grid→world mapping, and action masking
logic WITHOUT simulation — pure PyTorch tensor operations.
"""

import torch
import pytest


# ============================================================================
# Tests for Action Adapter (Blocker 1 Regression Tests)
# ============================================================================

def test_decoded_action_slot_propagation():
    """Prove continuous subsets properly decode to exact slots (e.g., slot 7)."""
    from pallet_rl.envs.action_adapter import decode_normalized_action
    
    # op_type(3), slot(10), x(16), y(24), rot(2)
    dims = (3, 10, 16, 24, 2)
    
    # For k=10, center of bin 7 is -1 + (2*7+1)/10 = 0.5
    action = torch.tensor([[0.0, 0.5, 0.0, 0.0, 0.0]])
    dec = decode_normalized_action(action, dims)
    assert dec.slot_idx[0].item() == 7


def test_buffer_uses_decoded_not_raw():
    """Prove retrieve coordinates decode to true grid indices."""
    from pallet_rl.envs.action_adapter import decode_normalized_action
    
    dims = (3, 10, 16, 24, 2)
    action = torch.tensor([[0.0, 0.0, -1.0, 0.95, 1.0]])
    
    dec = decode_normalized_action(action, dims)
    assert dec.grid_x[0].item() == 0
    assert dec.grid_y[0].item() == 23
    assert dec.rot_idx[0].item() == 1


def test_store_does_not_always_target_slot_0():
    """Test the original bug: float 0.9 was truncated to 0. It must be 9."""
    from pallet_rl.envs.action_adapter import decode_normalized_action
    
    dims = (3, 10, 16, 24, 2)
    action = torch.tensor([[0.0, 0.9, 0.0, 0.0, 0.0]])
    dec = decode_normalized_action(action, dims)
    assert dec.slot_idx[0].item() == 9


# ============================================================================
# Tests for AABB helpers (via usd_helpers)
# ============================================================================

def test_aabb_overlap_basic():
    """Overlapping and non-overlapping box pairs."""
    from pallet_rl.utils.usd_helpers import aabb_overlap

    pos_a = [0, 0, 0]
    dims_a = [1, 1, 1]

    # Fully overlapping
    assert aabb_overlap(pos_a, dims_a, [0, 0, 0], [1, 1, 1]) is True

    # Separated on X
    assert aabb_overlap(pos_a, dims_a, [2, 0, 0], [1, 1, 1]) is False

    # Touching (within margin → no overlap)
    assert aabb_overlap(pos_a, dims_a, [1.0, 0, 0], [1, 1, 1], margin=0.005) is False


def test_aabb_intersection_area():
    """2D intersection area computation."""
    from pallet_rl.utils.usd_helpers import aabb_intersection_area

    # Full overlap (both 1×1 at origin)
    area = aabb_intersection_area([0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1])
    assert abs(area - 1.0) < 1e-6

    # No overlap
    area = aabb_intersection_area([0, 0, 0], [1, 1, 1], [5, 5, 0], [1, 1, 1])
    assert area == 0.0

    # Partial overlap (0.5 in X, full in Y)
    area = aabb_intersection_area([0, 0, 0], [1, 1, 1], [0.5, 0, 0], [1, 1, 1])
    assert abs(area - 0.5) < 1e-6


# ============================================================================
# Tests for perception backend factory
# ============================================================================

def test_perception_factory_warp():
    """WarpBackend can be created via factory."""
    from pallet_rl.envs.perception import create_backend

    backend = create_backend("warp")
    assert backend.name == "warp"


def test_perception_factory_depth():
    """DepthCameraBackend can be created via factory."""
    from pallet_rl.envs.perception import create_backend

    backend = create_backend("depth_camera")
    assert backend.name == "depth_camera"


def test_perception_factory_invalid():
    """Factory raises ValueError for unknown backend name."""
    from pallet_rl.envs.perception import create_backend

    with pytest.raises(ValueError, match="Unknown heightmap backend"):
        create_backend("lidar")


if __name__ == "__main__":
    test_decode_action_basic()
    test_decode_action_roundtrip()
    test_to_discrete()
    test_to_discrete_uniform_bins()
    test_aabb_overlap_basic()
    test_aabb_intersection_area()
    test_perception_factory_warp()
    test_perception_factory_depth()
    test_perception_factory_invalid()
    test_decoded_action_slot_propagation()
    test_buffer_uses_decoded_not_raw()
    test_store_does_not_always_target_slot_0()
    print("\n✅ All modular tests passed!")
