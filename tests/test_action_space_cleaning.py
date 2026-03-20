"""
Focused unit tests for footprint-aware action space cleaning.
Does NOT require Isaac Lab.
"""

import torch
import pytest
from pallet_rl.envs.placement_controller import (
    _grid_to_pallet_center,
    _get_effective_xy_dims,
    _check_inside_pallet,
    _evaluate_patch_support,
    _validate_height_constraint,
    get_action_mask,
)

class MockCfg:
    def __init__(self):
        self.action_dims = (3, 10, 16, 24, 2)
        self.pallet_size = (1.2, 0.8)
        self.max_stack_height = 1.8
        self.map_shape = (160, 240)
        self.place_support_ratio_min = 0.60
        self.place_support_height_tol_m = 0.02
        self.place_border_epsilon_m = 1e-6
        self.max_boxes = 50
        self.buffer_slots = 10

class DecodedAction:
    def __init__(self, op_type, slot_idx, grid_x, grid_y, rot_idx):
        self.op_type = op_type
        self.slot_idx = slot_idx
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.rot_idx = rot_idx

class MockEnv:
    def __init__(self, num_envs=1, device="cpu"):
        self.num_envs = num_envs
        self._device = torch.device(device)
        self.cfg = MockCfg()
        self.box_idx = torch.zeros(num_envs, dtype=torch.long, device=self._device)
        self.box_dims = torch.zeros(num_envs, self.cfg.max_boxes, 3, device=self._device)
        self.buffer_state = torch.zeros(num_envs, self.cfg.buffer_slots, 6, device=self._device)
        self._height_invalid_mask = torch.zeros(num_envs, dtype=torch.bool, device=self._device)
        self._last_heightmap = None
        self.decoded_action = None

def test_grid_to_pallet_center():
    cfg = MockCfg()
    gx = torch.tensor([0, 15])
    gy = torch.tensor([0, 23])
    cx, cy = _grid_to_pallet_center(gx, gy, cfg.pallet_size, cfg.action_dims)
    
    # Pallet (1.2, 0.8), Grid (16, 24)
    # step_x = 1.2/16 = 0.075. half_x = 0.6. center(0) = 0*0.075 - 0.6 + 0.0375 = -0.5625
    # center(15) = 15*0.075 - 0.6 + 0.0375 = 1.125 - 0.6 + 0.0375 = 0.5625
    assert torch.allclose(cx, torch.tensor([-0.5625, 0.5625]))
    
    # step_y = 0.8/24 = 0.0333. half_y = 0.4. center(0) = -0.4 + 0.01666 = -0.38333
    assert torch.allclose(cy, torch.tensor([-0.3833333, 0.3833333]))

def test_rotation_aware_footprint():
    # Box (0.4, 0.3, 0.2)
    box_dims = torch.tensor([[0.4, 0.3, 0.2]])
    
    # rot_idx = 0 -> (0.4, 0.3)
    eff0 = _get_effective_xy_dims(box_dims, torch.tensor([0]))
    assert torch.allclose(eff0, torch.tensor([[0.4, 0.3]]))
    
    # rot_idx = 1 -> (0.3, 0.4)
    eff1 = _get_effective_xy_dims(box_dims, torch.tensor([1]))
    assert torch.allclose(eff1, torch.tensor([[0.3, 0.4]]))

def test_border_validity():
    cfg = MockCfg() # (1.2, 0.8)
    eff_xy = torch.tensor([[0.4, 0.3]])
    
    # Center (0, 0) -> Valid
    assert _check_inside_pallet(torch.tensor([0.0]), torch.tensor([0.0]), eff_xy, cfg.pallet_size).item() is True
    
    # Center (0.5, 0) -> 0.5 + 0.2 = 0.7 > 0.6 -> Invalid
    assert _check_inside_pallet(torch.tensor([0.5]), torch.tensor([0.0]), eff_xy, cfg.pallet_size).item() is False
    
    # Center (0.4, 0.25) -> 0.4+0.2=0.6, 0.25+0.15=0.4 -> Just on border -> Valid
    assert _check_inside_pallet(torch.tensor([0.4]), torch.tensor([0.25]), eff_xy, cfg.pallet_size).item() is True

def test_evaluate_patch_support():
    # Patch (H, W). Box H = 0.2. Max Stack = 1.8. Tol = 0.02. Min Ratio = 0.6
    
    # 1. Uniform support at 0 -> Valid
    patch = torch.zeros((10, 10))
    assert _evaluate_patch_support(patch, 0.2, 1.8, 0.02, 0.6).item() is True
    
    # 2. Exceeds max height
    patch = torch.ones((10, 10)) * 1.7
    assert _evaluate_patch_support(patch, 0.2, 1.8, 0.02, 0.6).item() is False
    
    # 3. Low support ratio (only 20% near top)
    patch = torch.zeros((10, 10))
    patch[:2, :2] = 0.1 # local_top = 0.1. threshold = 0.08. Only 4/100 cells >= 0.08.
    assert _evaluate_patch_support(patch, 0.2, 1.8, 0.02, 0.6).item() is False
    
    # 4. High support ratio (80% near top)
    patch = torch.zeros((10, 10))
    patch[:8, :] = 0.1 # 80/100 cells >= 0.08
    assert _evaluate_patch_support(patch, 0.2, 1.8, 0.02, 0.6).item() is True

def test_validate_height_constraint_footprint():
    env = MockEnv()
    env.box_dims[0, 0] = torch.tensor([0.4, 0.3, 0.2])
    env.decoded_action = DecodedAction(
        op_type=torch.tensor([0]), 
        slot_idx=torch.tensor([0]), 
        grid_x=torch.tensor([15]), # x=0.5625
        grid_y=torch.tensor([12]), # y=0.0166
        rot_idx=torch.tensor([0])
    )
    # Border check: cx=0.5625, dx=0.4 -> cx+dx/2 = 0.7625 > 0.6 -> Border fail
    
    env._last_heightmap = torch.zeros((1, 160, 240))
    
    _validate_height_constraint(
        env, 
        env.decoded_action.op_type, 
        env.decoded_action.slot_idx, 
        env.decoded_action.grid_x, 
        env.decoded_action.grid_y, 
        1, "cpu"
    )
    assert env._height_invalid_mask[0].item() is True

def test_conservative_action_mask_border():
    env = MockEnv()
    # Large box: (1.1, 0.7, 0.2)
    # Pallet: (1.2, 0.8)
    # Max X centers: 0.5625. cx+dx/2 = 0.5625 + 0.55 = 1.1125 > 0.6.
    # Min X centers: -0.5625. cx-dx/2 = -1.1125 < -0.6.
    # At center (cx=0), dx/2 = 0.55 < 0.6. OK.
    
    env.box_dims[0, 0] = torch.tensor([1.1, 0.7, 0.2])
    
    mask = get_action_mask(env)
    
    x_start = 3 + 10
    num_x = 16
    x_mask = mask[0, x_start : x_start + num_x]
    
    # Indices 0 and 15 should definitely be masked for a box this large
    assert x_mask[0].item() is False
    assert x_mask[15].item() is False
    # Center should be True
    assert x_mask[8].item() is True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
