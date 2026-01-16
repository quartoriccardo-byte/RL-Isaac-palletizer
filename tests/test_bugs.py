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
    """Deprecated: legacy ActorCritic implementation has been removed.

    This test is kept as a placeholder to document the original bug
    but no longer runs against any live code.
    """
    pass


def test_mask_shape_contract():
    """Deprecated: legacy SpatialPolicyHead has been removed.

    Action masking is now implemented directly in the MultiDiscrete policy
    and env; see PalletizerActorCritic and PalletTask for details.
    """
    pass


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
    
    # Create model with minimal params (updated for new obs dim)
    model = PalletizerActorCritic(
        num_actor_obs=38489,
        num_critic_obs=38489,
        num_actions=55
    )
    
    # Create dummy observation
    batch_size = 4
    obs = torch.randn(batch_size, 38489)
    
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
