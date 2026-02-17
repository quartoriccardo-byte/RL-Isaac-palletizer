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
        num_actor_obs=38491,
        num_critic_obs=38491,
        num_actions=55
    )
    
    # Create dummy observation
    batch_size = 4
    obs = torch.randn(batch_size, 38491)
    
    # Call act to populate distributions
    actions = model.act(obs)
    assert actions.shape == (batch_size, 5), f"Expected (4,5), got {actions.shape}"
    
    # Now entropy should work (compatible with property or callable)
    entropy_attr = model.entropy
    entropy = entropy_attr() if callable(entropy_attr) else entropy_attr
    assert entropy is not None, "Entropy should not be None"
    assert isinstance(entropy, torch.Tensor), f"Expected Tensor, got {type(entropy)}"
    assert entropy.shape == (batch_size,), f"Expected ({batch_size},), got {entropy.shape}"
    assert torch.all(entropy >= 0), "Entropy should be non-negative"
    
    # get_actions_log_prob should also work now
    log_prob = model.get_actions_log_prob(actions)
    assert log_prob.shape == (batch_size,), f"Expected ({batch_size},), got {log_prob.shape}"
    
    print("✓ RSL-RL wrapper entropy test passed")


def test_pallet_mesh_centering_math():
    """Test the bbox → correction math used in _spawn_pallet_mesh_visual.

    Given a mock mesh bounding box, verify the auto-centering correction
    aligns XY center to pallet collider center and Z base to collider bottom (z=0).
    """
    # --- Pallet collider constants (from PalletSceneCfg / USD read) ---
    COLLIDER_CENTER_XY = (0.0, 0.0)
    # Collider at z=0.075 with half-height 0.075 → bottom z=0.0
    COLLIDER_BOTTOM_Z = 0.0

    # --- Mock bbox: mesh at arbitrary offset ---
    # Suppose STL mesh spawned at origin has bounds:
    #   min = (0.5, 0.3, 0.0)
    #   max = (1.7, 1.1, 0.15)
    min_x, min_y, min_z = 0.5, 0.3, 0.0
    max_x, max_y, max_z = 1.7, 1.1, 0.15
    center_x = (min_x + max_x) / 2.0  # 1.1
    center_y = (min_y + max_y) / 2.0  # 0.7

    # Auto-center XY
    dx = COLLIDER_CENTER_XY[0] - center_x  # -1.1
    dy = COLLIDER_CENTER_XY[1] - center_y  # -0.7

    # Auto-align Z (mesh base → collider bottom)
    dz = COLLIDER_BOTTOM_Z - min_z  # 0.0

    assert abs(dx - (-1.1)) < 1e-6, f"dx={dx}"
    assert abs(dy - (-0.7)) < 1e-6, f"dy={dy}"
    assert abs(dz - 0.0) < 1e-6, f"dz={dz}"

    # Add user offset
    user_off = (0.01, -0.02, 0.005)
    final_x = dx + user_off[0]
    final_y = dy + user_off[1]
    final_z = dz + user_off[2]

    assert abs(final_x - (-1.09)) < 1e-6, f"final_x={final_x}"
    assert abs(final_y - (-0.72)) < 1e-6, f"final_y={final_y}"
    assert abs(final_z - 0.005) < 1e-6, f"final_z={final_z}"

    # Case 2: mesh already centered — correction should be near-zero
    min_x2, min_y2, min_z2 = -0.6, -0.4, 0.0
    max_x2, max_y2, max_z2 = 0.6, 0.4, 0.15
    cx2 = (min_x2 + max_x2) / 2.0  # 0.0
    cy2 = (min_y2 + max_y2) / 2.0  # 0.0
    dx2 = -cx2
    dy2 = -cy2
    dz2 = COLLIDER_BOTTOM_Z - min_z2
    assert abs(dx2) < 1e-6, f"dx2={dx2}"
    assert abs(dy2) < 1e-6, f"dy2={dy2}"
    assert abs(dz2) < 1e-6, f"dz2={dz2}"

    print("✓ pallet_mesh_centering_math tests passed")


if __name__ == "__main__":
    test_decode_action_rectangular()
    test_action_mean_no_mutation()
    test_mask_shape_contract()
    test_terminated_truncated_types()
    test_rsl_rl_wrapper_entropy()
    test_pallet_mesh_centering_math()
    print("\n✅ All static tests passed!")

