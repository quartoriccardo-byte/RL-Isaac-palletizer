
import yaml
import torch

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def decode_action(action_idx: torch.Tensor, width: int, num_rotations: int):
    """
    Decodes a flat action index into (rot_idx, x, y) coordinates.
    Formula: index = rot * (W*H) + y * W + x
    """
    area = width * width
    
    # 1. Extract Rotation
    rot_idx = action_idx // area
    
    # 2. Extract Spatial Position
    spatial_idx = action_idx % area
    y = spatial_idx // width
    x = spatial_idx % width
    
    return rot_idx, x, y
