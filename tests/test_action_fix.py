"""
Unit tests for the centralized action contract.

These tests run WITHOUT Isaac Lab — pure PyTorch.
Tests verify:
- `is_discrete_action_tensor` correctly identifies discrete vs normalized.
- `[[1,1,1,1,1]]` is handled as discrete.
- Ambiguous or out-of-range inputs raise `ValueError`.
- `decode_action_tensor` returns correct `DecodedAction` fields.
"""

import pytest
import torch
from pallet_rl.envs.action_adapter import (
    is_discrete_action_tensor,
    decode_action_tensor,
    discrete_to_center_normalized
)

ACTION_DIMS = (3, 10, 16, 24, 2)

class TestActionContract:
    def test_discrete_int_tensor(self):
        """Integer tensors are always discrete."""
        actions = torch.tensor([[0, 5, 8, 12, 0]], dtype=torch.long)
        assert is_discrete_action_tensor(actions, ACTION_DIMS) is True
        
        dec = decode_action_tensor(actions, ACTION_DIMS)
        assert dec.op_type.item() == 0
        assert dec.slot_idx.item() == 5

    def test_discrete_float_integral_tensor(self):
        """Float tensors with integral values in-range are discrete."""
        actions = torch.tensor([[2.0, 9.0, 15.0, 23.0, 1.0]], dtype=torch.float32)
        assert is_discrete_action_tensor(actions, ACTION_DIMS) is True
        
        dec = decode_action_tensor(actions, ACTION_DIMS)
        assert dec.op_type.item() == 2
        assert dec.grid_y.item() == 23

    def test_regression_ones_tensor(self):
        """[[1,1,1,1,1]] must be handled as discrete, not normalized."""
        actions = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
        assert is_discrete_action_tensor(actions, ACTION_DIMS) is True
        
        dec = decode_action_tensor(actions, ACTION_DIMS)
        # In the old heuristic, this was read as normalized op_type=0 (Place).
        # In the new contract, it's discrete op_type=1 (Store).
        assert dec.op_type.item() == 1
        assert dec.slot_idx.item() == 1

    def test_normalized_float_tensor(self):
        """Float tensors in [-1, 1] with non-integral values are normalized."""
        actions = torch.tensor([[0.5, -0.2, 0.0, 0.8, -1.0]], dtype=torch.float32)
        assert is_discrete_action_tensor(actions, ACTION_DIMS) is False
        
        dec = decode_action_tensor(actions, ACTION_DIMS)
        # op_dims=3, 0.5 -> bin 2 (Retrieve)
        # _to_discrete(0.5, 3) = floor((0.5+1)/2 * 3) = floor(0.75 * 3) = floor(2.25) = 2
        assert dec.op_type.item() == 2

    def test_invalid_ambiguous_tensor(self):
        """Non-integral values outside [-1, 1] must raise ValueError."""
        # Value 2.5 is not integral and outside [-1, 1]
        actions = torch.tensor([[2.5, 0, 0, 0, 0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="Ambiguous or invalid action tensor"):
            decode_action_tensor(actions, ACTION_DIMS)

    def test_invalid_range_discrete(self):
        """Integral values outside [0, k-1] are not discrete and if outside [-1, 1] are invalid."""
        # op=5 is integral but outside [0, 2]
        actions = torch.tensor([[5.0, 0, 0, 0, 0]], dtype=torch.float32)
        # It's not discrete (out of range) and not normalized (> 1.0)
        with pytest.raises(ValueError, match="Ambiguous or invalid action tensor"):
            decode_action_tensor(actions, ACTION_DIMS)

    def test_discrete_to_normalized_roundtrip(self):
        """Test that discrete_to_center_normalized produces values that decode back correctly."""
        original = torch.tensor([[0, 9, 15, 23, 1]], dtype=torch.long)
        norm = discrete_to_center_normalized(original, ACTION_DIMS)
        
        assert torch.all(norm >= -1.0)
        assert torch.all(norm <= 1.0)
        
        # Should NOT be discrete because they are center-of-bin floats (e.g. -0.666)
        assert is_discrete_action_tensor(norm, ACTION_DIMS) is False
        
        dec = decode_action_tensor(norm, ACTION_DIMS)
        assert dec.op_type.item() == 0
        assert dec.slot_idx.item() == 9
        assert dec.grid_x.item() == 15
        assert dec.grid_y.item() == 23
        assert dec.rot_idx.item() == 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
