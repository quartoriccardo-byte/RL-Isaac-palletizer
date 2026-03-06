"""
Unit tests for the extracted placement controller module.

These tests validate action decoding, grid→world mapping, and action masking
logic WITHOUT simulation — pure PyTorch tensor operations.
"""

import torch
import pytest


# ============================================================================
# Tests for decode_action (legacy flat-index decoder)
# ============================================================================

def test_decode_action_basic():
    """Test legacy flat-index → (rot, x, y) decoding."""
    from pallet_rl.envs.placement_controller import decode_action

    # index = rot * (H * W) + y * W + x
    W, H, R = 16, 24, 2

    # First cell: rot=0, y=0, x=0
    rot, x, y = decode_action(0, W, H, R)
    assert rot == 0 and x == 0 and y == 0

    # Last cell of first rotation: rot=0, y=23, x=15
    rot, x, y = decode_action(H * W - 1, W, H, R)
    assert rot == 0 and x == W - 1 and y == H - 1

    # First cell of second rotation: rot=1, y=0, x=0
    rot, x, y = decode_action(H * W, W, H, R)
    assert rot == 1 and x == 0 and y == 0


def test_decode_action_roundtrip():
    """Encode an index and decode it back — should be consistent."""
    from pallet_rl.envs.placement_controller import decode_action

    W, H, R = 16, 24, 2
    for r in range(R):
        for y_in in range(H):
            for x_in in range(W):
                idx = r * (H * W) + y_in * W + x_in
                rot_out, x_out, y_out = decode_action(idx, W, H, R)
                assert rot_out == r, f"rot mismatch at idx={idx}"
                assert x_out == x_in, f"x mismatch at idx={idx}"
                assert y_out == y_in, f"y mismatch at idx={idx}"


# ============================================================================
# Tests for _to_discrete
# ============================================================================

def test_to_discrete():
    """Map continuous [-1, 1] → discrete {0..k-1}."""
    from pallet_rl.envs.placement_controller import _to_discrete

    # -1 → 0, 1 → k-1
    k = 10
    a = torch.tensor([-1.0, 0.0, 1.0])
    result = _to_discrete(a, k)
    assert result[0].item() == 0
    assert result[1].item() == 5  # center
    assert result[2].item() == k - 1  # edge case: 1.0 maps to 9


def test_to_discrete_uniform_bins():
    """Uniform sampling of [-1, 1] should produce roughly uniform bins."""
    from pallet_rl.envs.placement_controller import _to_discrete

    k = 4
    # Sample center of each bin
    centers = torch.tensor([-0.75, -0.25, 0.25, 0.75])
    result = _to_discrete(centers, k)
    assert result.tolist() == [0, 1, 2, 3]


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
    print("\n✅ All modular tests passed!")
