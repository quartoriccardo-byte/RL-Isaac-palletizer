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


def test_infeasibility_termination():
    """Test infeasibility detection using num_boxes (episode allocation)."""
    n = 4
    max_payload_kg = 500.0
    num_boxes = 40  # Episode allocation
    base_box_mass_kg = 10.0
    
    # Env 0: 100kg on pallet, 20 boxes placed. Remaining: (40-20)*10 = 200. Total 300. -> OK
    # Env 1: 400kg on pallet, 35 boxes placed. Remaining: (40-35)*10 = 50. Total 450. -> OK
    # Env 2: 450kg on pallet, 30 boxes placed. Remaining: (40-30)*10 = 100. Total 550. -> INFEASIBLE
    # Env 3: 100kg on pallet, 40 boxes placed, BUT 50kg in buffer. Remaining: (40-40)*10 = 0. Total 150. -> OK
    
    payload_kg = torch.tensor([100.0, 400.0, 450.0, 100.0])
    box_idx = torch.tensor([20, 35, 30, 40])
    buffer_state = torch.zeros(n, 10, 6)
    buffer_has_box = torch.zeros(n, 10, dtype=torch.bool)
    
    buffer_state[3, 0, 5] = 50.0
    buffer_has_box[3, 0] = True
    
    # Regression: Logic must use num_boxes, not max_boxes
    remaining_boxes = (num_boxes - box_idx).clamp(min=0).float()
    remaining_mass = remaining_boxes * base_box_mass_kg
    buffer_mass = (buffer_state[:, :, 5] * buffer_has_box.float()).sum(dim=1)
    
    prospective_total = payload_kg + buffer_mass + remaining_mass
    infeasible_mask = prospective_total > max_payload_kg
    
    expected = torch.tensor([False, False, True, False])
    assert (infeasible_mask == expected).all(), \
        f"Infeasible mask {infeasible_mask} != expected {expected}\nTotals: {prospective_total}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
