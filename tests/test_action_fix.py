"""
Unit tests for the action space fix (discrete → center-of-bin conversion).

These tests run WITHOUT Isaac Lab — pure PyTorch.
Tests verify:
- Discrete indices are correctly converted to normalized [-1, 1] values
- Already-normalized float actions pass through unchanged
- Round-trip: discrete → normalized → decode recovers original indices
"""

import pytest
import torch


# Replicate the conversion logic from pallet_task._pre_physics_step
ACTION_DIMS = (3, 10, 16, 24, 2)  # Op, Slot, X, Y, Rot


def discrete_to_normalized(actions: torch.Tensor, dims: tuple[int, ...] = ACTION_DIMS) -> torch.Tensor:
    """
    Convert discrete indices to center-of-bin normalized values.
    
    Replicates the logic in PalletTask._pre_physics_step.
    """
    actions = actions.float()
    is_discrete = actions.abs().max() > 1.5
    
    if is_discrete:
        for col, k in enumerate(dims):
            actions[:, col] = (actions[:, col] + 0.5) / k * 2.0 - 1.0
    
    return torch.clamp(actions, -1.0, 1.0)


def normalized_to_discrete(actions: torch.Tensor, dims: tuple[int, ...] = ACTION_DIMS) -> torch.Tensor:
    """
    Convert normalized [-1, 1] values back to discrete indices.
    
    Replicates the decoding logic in PalletTask._apply_action.
    """
    indices = []
    for col, k in enumerate(dims):
        # Map [-1, 1] → [0, K-1]
        idx = ((actions[:, col] + 1.0) / 2.0 * k).floor().clamp(0, k - 1).long()
        indices.append(idx)
    return torch.stack(indices, dim=-1)


class TestDiscreteConversion:
    """Test discrete → normalized conversion."""

    def test_basic_conversion(self):
        """Verify specific indices convert to expected normalized values."""
        # Note: detection threshold is max > 1.5, so we need at least one value > 1.5
        # Use a realistic multi-env batch where discrete indices are mixed
        actions = torch.tensor([[2, 5, 8, 12, 1]], dtype=torch.float32)
        result = discrete_to_normalized(actions.clone())
        # For Op=2, K=3: (2 + 0.5) / 3 * 2 - 1 = 5/3 - 1 = 2/3 ≈ 0.6667
        expected_op = (2 + 0.5) / 3 * 2.0 - 1.0
        assert abs(result[0, 0].item() - expected_op) < 1e-5

    def test_max_indices(self):
        """Verify max valid indices stay within [-1, 1]."""
        max_indices = torch.tensor([[2, 9, 15, 23, 1]], dtype=torch.float32)
        result = discrete_to_normalized(max_indices.clone())
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_all_values_in_range(self):
        """All converted values must be in [-1, 1]."""
        # Test all possible index combinations for a batch
        actions = torch.tensor([
            [0, 0, 0, 0, 0],
            [1, 5, 8, 12, 0],
            [2, 9, 15, 23, 1],
            [0, 3, 7, 20, 1],
        ], dtype=torch.float32)
        result = discrete_to_normalized(actions.clone())
        assert (result >= -1.0).all()
        assert (result <= 1.0).all()

    def test_previously_broken_indices(self):
        """
        Indices > 1 were previously clamped to 1.0, losing all information.
        After fix, each should map to a unique bin.
        """
        # Slot indices 0..9 should all produce different normalized values
        actions = torch.zeros(10, 5, dtype=torch.float32)
        for i in range(10):
            actions[i, 1] = float(i)  # slot column
        
        result = discrete_to_normalized(actions.clone())
        slot_values = result[:, 1]
        
        # All should be unique
        assert slot_values.unique().shape[0] == 10, \
            f"Expected 10 unique slot values, got {slot_values.unique().shape[0]}"


class TestPassthrough:
    """Already-normalized float actions should pass through unchanged."""

    def test_float_actions_unchanged(self):
        """Actions in [-1, 1] should not be modified (passthrough)."""
        actions = torch.tensor([
            [0.5, -0.3, 0.0, 0.8, -0.9],
            [-1.0, 1.0, 0.0, -0.5, 0.5],
        ], dtype=torch.float32)
        result = discrete_to_normalized(actions.clone())
        assert torch.allclose(result, actions.clamp(-1, 1))

    def test_detection_threshold(self):
        """Actions with all values ≤ 1.5 are treated as normalized floats."""
        actions = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
        result = discrete_to_normalized(actions.clone())
        assert torch.allclose(result, actions)  # passthrough

    def test_detection_triggers_above_threshold(self):
        """Actions with any value > 1.5 trigger discrete conversion."""
        actions = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        result = discrete_to_normalized(actions.clone())
        # Op=2 should NOT equal 1.0 after conversion
        assert result[0, 0].item() != 1.0  # was converted, not clamped


class TestRoundTrip:
    """Discrete → normalized → decode should recover original indices."""

    def test_roundtrip_basic(self):
        """Simple indices should round-trip correctly."""
        original = torch.tensor([[1, 5, 8, 12, 0]], dtype=torch.float32)
        normalized = discrete_to_normalized(original.clone())
        recovered = normalized_to_discrete(normalized)
        
        expected = original.long()
        assert torch.equal(recovered, expected), \
            f"Round-trip failed: {original} → {normalized} → {recovered}"

    def test_roundtrip_all_extremes(self):
        """Min and max indices should round-trip when batched together."""
        # Batch them together so max > 1.5 triggers discrete detection
        combined = torch.tensor([
            [0, 0, 0, 0, 0],   # min indices
            [2, 9, 15, 23, 1], # max indices
        ], dtype=torch.float32)
        
        normalized = discrete_to_normalized(combined.clone())
        recovered = normalized_to_discrete(normalized)
        assert torch.equal(recovered, combined.long()), \
            f"Round-trip failed: {combined} → {normalized} → {recovered}"

    def test_roundtrip_batch(self):
        """Batch of various indices should all round-trip."""
        originals = torch.tensor([
            [0, 0, 0, 0, 0],
            [1, 3, 7, 11, 1],
            [2, 9, 15, 23, 0],
            [0, 5, 10, 18, 1],
            [2, 7, 3, 22, 0],
        ], dtype=torch.float32)
        
        normalized = discrete_to_normalized(originals.clone())
        recovered = normalized_to_discrete(normalized)
        assert torch.equal(recovered, originals.long()), \
            f"Batch round-trip failed:\n  Original: {originals}\n  Recovered: {recovered}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
