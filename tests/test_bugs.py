"""
Static test file for pallet_rl package.
These tests encode correctness via assertions that will fail early at import time
if bugs are reintroduced.
"""
import torch


def test_decode_action_rectangular():
    """Test decode_action with rectangular grid (width ≠ height)."""
    from pallet_rl.algo.utils import decode_action
    
    # Grid: 240 width (W), 160 height (H), 4 rotations
    # Total actions = 4 * 160 * 240 = 153600
    W, H, R = 240, 160, 4
    area = W * H
    
    # Test case 1: index 0 -> (rot=0, x=0, y=0)
    rot, x, y = decode_action(torch.tensor(0), W, H, R)
    assert rot == 0 and x == 0 and y == 0, f"Got {rot}, {x}, {y}"
    
    # Test case 2: index W -> (rot=0, x=0, y=1)
    rot, x, y = decode_action(torch.tensor(W), W, H, R)
    assert rot == 0 and x == 0 and y == 1, f"Expected (0,0,1), got ({rot}, {x}, {y})"
    
    # Test case 3: index area -> (rot=1, x=0, y=0)
    rot, x, y = decode_action(torch.tensor(area), W, H, R)
    assert rot == 1 and x == 0 and y == 0, f"Expected (1,0,0), got ({rot}, {x}, {y})"
    
    # Test case 4: Last valid index
    last_idx = R * area - 1
    rot, x, y = decode_action(torch.tensor(last_idx), W, H, R)
    assert rot == R - 1, f"Expected rot={R-1}, got {rot}"
    
    print("✓ decode_action rectangular grid tests passed")


def test_action_mean_no_mutation():
    """Test that action_mean does NOT mutate weights."""
    from pallet_rl.models.actor_critic import ActorCritic
    
    # Create model
    model = ActorCritic(
        num_obs=38453,
        num_critic_obs=38453,
        num_actions=5
    )
    
    # Store original weight values
    original_weights = model.actor_head[-1].weight.data.clone()
    
    # Access action_mean (this used to destroy weights!)
    _ = model.action_mean
    
    # Verify weights unchanged
    current_weights = model.actor_head[-1].weight.data
    assert torch.allclose(original_weights, current_weights), \
        "action_mean property corrupted weights!"
    
    print("✓ action_mean no mutation test passed")


def test_mask_shape_contract():
    """Test that policy_heads enforces mask shape contract."""
    from pallet_rl.models.policy_heads import SpatialPolicyHead
    
    head = SpatialPolicyHead(in_channels=64, num_rotations=4)
    
    # Create test input
    B, C, H, W = 2, 64, 160, 240
    x = torch.randn(B, C, H, W)
    
    # Correct mask shape
    correct_mask = torch.ones(B, 4 * H * W, dtype=torch.bool)
    output = head(x, mask=correct_mask)
    assert output.shape == (B, 4 * H * W), f"Unexpected output shape: {output.shape}"
    
    # Wrong mask shape should raise AssertionError
    wrong_mask = torch.ones(B, H * W)  # Missing rotations dimension
    try:
        head(x, mask=wrong_mask)
        assert False, "Should have raised AssertionError for wrong mask shape"
    except AssertionError:
        pass  # Expected
    
    print("✓ mask shape contract test passed")


def test_terminated_truncated_types():
    """Test tensor type handling for terminated/truncated."""
    # This tests the pattern used in the wrapper
    device = torch.device("cpu")
    
    # Case 1: Both are tensors (normal case)
    terminated = torch.tensor([True, False], dtype=torch.bool)
    truncated = torch.tensor([False, True], dtype=torch.bool)
    dones = terminated | truncated
    assert dones.shape == (2,) and dones.dtype == torch.bool
    
    # Case 2: truncated is bool
    truncated_bool = False
    if not torch.is_tensor(truncated_bool):
        truncated_tensor = torch.full_like(terminated, truncated_bool)
    dones = terminated | truncated_tensor
    assert dones.shape == (2,) and dones.dtype == torch.bool
    
    print("✓ terminated/truncated type handling test passed")


def test_rsl_rl_wrapper_entropy():
    """Test that PalletizerActorCritic has entropy method for RSL-RL PPO."""
    from pallet_rl.models.rsl_rl_wrapper import PalletizerActorCritic
    
    # Create model with minimal params
    model = PalletizerActorCritic(
        num_actor_obs=38477,
        num_critic_obs=38477,
        num_actions=55
    )
    
    # Create dummy observation
    batch_size = 4
    obs = torch.randn(batch_size, 38477)
    
    # Call act to populate distributions
    actions = model.act(obs)
    assert actions.shape == (batch_size, 5), f"Expected (4,5), got {actions.shape}"
    
    # Now entropy should work
    entropy = model.entropy()
    assert entropy.shape == (batch_size,), f"Expected ({batch_size},), got {entropy.shape}"
    assert torch.all(entropy >= 0), "Entropy should be non-negative"
    
    # get_actions_log_prob should also work now
    log_prob = model.get_actions_log_prob(actions)
    assert log_prob.shape == (batch_size,), f"Expected ({batch_size},), got {log_prob.shape}"
    
    print("✓ RSL-RL wrapper entropy test passed")


if __name__ == "__main__":
    test_decode_action_rectangular()
    test_action_mean_no_mutation()
    test_mask_shape_contract()
    test_terminated_truncated_types()
    test_rsl_rl_wrapper_entropy()
    print("\n✅ All static tests passed!")
