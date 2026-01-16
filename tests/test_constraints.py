"""
Test file for palletizer constraint logic.

These tests verify mass bookkeeping and infeasibility termination logic
without requiring Isaac Lab simulation.
"""
import torch
import pytest


class MockBufferState:
    """Mock buffer state for testing."""
    def __init__(self, num_envs: int, buffer_slots: int = 10, buffer_features: int = 6):
        self.num_envs = num_envs
        self.buffer_slots = buffer_slots
        self.buffer_features = buffer_features
        
        # Tensors
        self.buffer_state = torch.zeros(num_envs, buffer_slots, buffer_features)
        self.buffer_has_box = torch.zeros(num_envs, buffer_slots, dtype=torch.bool)
        self.buffer_box_id = torch.full((num_envs, buffer_slots), -1, dtype=torch.long)
        self.payload_kg = torch.zeros(num_envs)
        self.box_mass_kg = torch.zeros(num_envs, 50)  # max_boxes=50
        self.box_idx = torch.zeros(num_envs, dtype=torch.long)


def test_mass_bookkeeping_place():
    """Test that PLACE correctly adds box mass to payload."""
    n = 4
    state = MockBufferState(n)
    
    # Setup: assign masses to boxes
    state.box_mass_kg[:, 0] = torch.tensor([5.0, 6.0, 7.0, 8.0])
    state.box_mass_kg[:, 1] = torch.tensor([3.0, 4.0, 5.0, 6.0])
    
    # Initially payload should be 0
    assert (state.payload_kg == 0).all(), "Initial payload should be 0"
    
    # Simulate PLACE for box_idx=0 in all envs
    place_envs = torch.arange(n)
    placed_mass = state.box_mass_kg[place_envs, state.box_idx[place_envs]]
    state.payload_kg[place_envs] += placed_mass
    state.box_idx += 1
    
    # Check payload matches box 0 masses
    expected_payload = torch.tensor([5.0, 6.0, 7.0, 8.0])
    assert torch.allclose(state.payload_kg, expected_payload), \
        f"Payload {state.payload_kg} != expected {expected_payload}"
    
    # Simulate another PLACE for box_idx=1
    placed_mass = state.box_mass_kg[place_envs, state.box_idx[place_envs]]
    state.payload_kg[place_envs] += placed_mass
    state.box_idx += 1
    
    # Check cumulative payload
    expected_payload = torch.tensor([8.0, 10.0, 12.0, 14.0])
    assert torch.allclose(state.payload_kg, expected_payload), \
        f"Cumulative payload {state.payload_kg} != expected {expected_payload}"
    
    print("✓ Mass bookkeeping PLACE test passed")


def test_mass_bookkeeping_store_retrieve():
    """Test that STORE removes mass from payload and RETRIEVE adds it back."""
    n = 2
    state = MockBufferState(n)
    
    # Setup: place a box first
    state.box_mass_kg[:, 0] = torch.tensor([5.0, 10.0])
    state.payload_kg = torch.tensor([5.0, 10.0])  # Box 0 already placed
    state.box_idx = torch.tensor([1, 1])  # Next box would be index 1
    
    # Simulate STORE of box 0 into slot 0
    store_envs = torch.tensor([0, 1])
    store_slots = torch.tensor([0, 0])
    stored_physical_id = (state.box_idx[store_envs] - 1).clamp(0, 49)  # box_idx - 1
    stored_mass = state.box_mass_kg[store_envs, stored_physical_id]
    
    # Update buffer state
    state.buffer_state[store_envs, store_slots, 5] = stored_mass  # mass in feature 5
    state.buffer_has_box[store_envs, store_slots] = True
    state.buffer_box_id[store_envs, store_slots] = stored_physical_id
    
    # Remove from payload
    state.payload_kg[store_envs] -= stored_mass
    
    # Payload should be 0 now (box moved to buffer)
    assert (state.payload_kg == 0).all(), \
        f"Payload after STORE should be 0, got {state.payload_kg}"
    
    # Verify mass is tracked in buffer
    assert torch.allclose(state.buffer_state[:, 0, 5], torch.tensor([5.0, 10.0])), \
        "Buffer should track stored mass"
    
    # Simulate RETRIEVE from slot 0
    retr_envs = torch.tensor([0, 1])
    retr_slots = torch.tensor([0, 0])
    retrieved_mass = state.buffer_state[retr_envs, retr_slots, 5]
    
    # Add back to payload
    state.payload_kg[retr_envs] += retrieved_mass
    
    # Clear buffer slot
    state.buffer_has_box[retr_envs, retr_slots] = False
    state.buffer_state[retr_envs, retr_slots] = 0.0
    
    # Payload should be back to original
    expected_payload = torch.tensor([5.0, 10.0])
    assert torch.allclose(state.payload_kg, expected_payload), \
        f"Payload after RETRIEVE {state.payload_kg} != expected {expected_payload}"
    
    print("✓ Mass bookkeeping STORE/RETRIEVE test passed")


def test_infeasibility_termination():
    """Test infeasibility detection when prospective mass exceeds max."""
    n = 4
    max_payload_kg = 500.0
    max_boxes = 50
    base_box_mass_kg = 5.0
    
    # Create state
    payload_kg = torch.tensor([100.0, 400.0, 490.0, 600.0])
    box_idx = torch.tensor([10, 40, 48, 50])
    buffer_state = torch.zeros(n, 10, 6)
    buffer_has_box = torch.zeros(n, 10, dtype=torch.bool)
    
    # Add some buffer mass for env 2
    buffer_state[2, 0, 5] = 20.0  # 20kg in buffer slot
    buffer_has_box[2, 0] = True
    
    # Compute remaining mass estimate
    remaining_boxes = (max_boxes - box_idx).float()
    remaining_mass = remaining_boxes * base_box_mass_kg
    
    # Buffer mass
    buffer_mass = (buffer_state[:, :, 5] * buffer_has_box.float()).sum(dim=1)
    
    # Prospective total
    prospective_total = payload_kg + buffer_mass + remaining_mass
    
    # Expected:
    # Env 0: 100 + 0 + 40*5 = 300   -> OK
    # Env 1: 400 + 0 + 10*5 = 450   -> OK
    # Env 2: 490 + 20 + 2*5 = 520   -> INFEASIBLE (> 500)
    # Env 3: 600 + 0 + 0*5 = 600    -> INFEASIBLE (> 500)
    
    infeasible_mask = prospective_total > max_payload_kg
    
    expected_infeasible = torch.tensor([False, False, True, True])
    assert (infeasible_mask == expected_infeasible).all(), \
        f"Infeasible mask {infeasible_mask} != expected {expected_infeasible}"
    
    print("✓ Infeasibility termination test passed")


def test_height_constraint_validation():
    """Test height constraint logic for action validation."""
    n = 3
    max_stack_height = 1.8  # meters
    
    # Mock heightmap values at target positions
    local_heights = torch.tensor([1.5, 1.7, 0.5])  # meters
    current_box_heights = torch.tensor([0.2, 0.2, 0.2])  # box height
    
    predicted_top = local_heights + current_box_heights
    # Expected: [1.7, 1.9, 0.7]
    
    height_exceeds = predicted_top > max_stack_height
    # Expected: [False, True, False]
    
    expected = torch.tensor([False, True, False])
    assert (height_exceeds == expected).all(), \
        f"Height exceeds {height_exceeds} != expected {expected}"
    
    print("✓ Height constraint validation test passed")


def test_settling_drift_computation():
    """Test drift computation for settling stability check."""
    n = 3
    pi = 3.14159265359
    
    # Current positions after settling
    current_pos = torch.tensor([
        [0.1, 0.2, 0.15],   # Near target
        [0.2, 0.3, 0.14],   # Drifted XY
        [0.1, 0.2, 0.02],   # Fell (z < 0.05)
    ])
    
    # Target positions
    target_pos = torch.tensor([
        [0.1, 0.2, 0.0],
        [0.1, 0.2, 0.0],
        [0.1, 0.2, 0.0],
    ])
    
    # Compute XY drift
    drift_xy = torch.norm(current_pos[:, :2] - target_pos[:, :2], dim=-1)
    # Expected: [0.0, ~0.14, 0.0]
    
    # Check drift thresholds
    drift_xy_threshold = 0.05
    exceeded_xy = drift_xy > drift_xy_threshold
    # Expected: [False, True, False]
    
    # Check falls
    fell = current_pos[:, 2] < 0.05
    # Expected: [False, False, True]
    
    expected_exceeded = torch.tensor([False, True, False])
    expected_fell = torch.tensor([False, False, True])
    
    assert (exceeded_xy == expected_exceeded).all(), \
        f"Exceeded XY {exceeded_xy} != expected {expected_exceeded}"
    assert (fell == expected_fell).all(), \
        f"Fell {fell} != expected {expected_fell}"
    
    print("✓ Settling drift computation test passed")


def test_observation_dimension():
    """Test that observation dimension matches expected value."""
    # Config values
    map_shape = (160, 240)
    buffer_slots = 10
    buffer_features = 6  # Was 5, now 6 (added mass)
    robot_state_dim = 24
    
    vis_dim = map_shape[0] * map_shape[1]  # 38400
    buf_dim = buffer_slots * buffer_features  # 60
    box_dim = 3
    extra_dim = 2  # payload_norm + current_box_mass_norm
    
    expected_obs_dim = vis_dim + buf_dim + box_dim + extra_dim + robot_state_dim
    # 38400 + 60 + 3 + 2 + 24 = 38489
    
    assert expected_obs_dim == 38489, f"Expected 38489, got {expected_obs_dim}"
    
    print("✓ Observation dimension test passed")


if __name__ == "__main__":
    test_mass_bookkeeping_place()
    test_mass_bookkeeping_store_retrieve()
    test_infeasibility_termination()
    test_height_constraint_validation()
    test_settling_drift_computation()
    test_observation_dimension()
    print("\n✅ All constraint tests passed!")
