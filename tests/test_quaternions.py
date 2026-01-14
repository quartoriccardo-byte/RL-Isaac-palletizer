"""
Unit tests for quaternion conversion helpers.
These are simulation-free checks that validate our convention choices.
"""

from __future__ import annotations

import torch

from pallet_rl.utils.quaternions import wxyz_to_xyzw, xyzw_to_wxyz, is_unit_quaternion


def test_roundtrip_conversion():
    """wxyz -> xyzw -> wxyz should be identity."""
    q_wxyz = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9238795, 0.0, 0.0, 0.3826834],
        ]
    )
    q_xyzw = wxyz_to_xyzw(q_wxyz)
    q_back = xyzw_to_wxyz(q_xyzw)
    assert torch.allclose(q_back, q_wxyz, atol=1e-6)


def test_is_unit_quaternion():
    """Detect unit vs. non-unit quaternions."""
    q_unit = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q_scaled = 2.0 * q_unit

    is_unit = is_unit_quaternion(q_unit)
    is_unit_scaled = is_unit_quaternion(q_scaled)

    assert bool(is_unit.item()) is True
    assert bool(is_unit_scaled.item()) is False


if __name__ == "__main__":
    test_roundtrip_conversion()
    test_is_unit_quaternion()
    print("\n✅ Quaternion helper tests passed!")

