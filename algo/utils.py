
import yaml

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def decode_action(action_idx, width, num_rotations):
    """
    Decodes a flat action index into (rot_idx, x, y).
    Args:
        action_idx: Tensor of shape (Batch,)
        width: Width of the heightmap (e.g., 128)
        num_rotations: Number of discrete rotations (e.g., 4)
    """
    area = width * width
    
    # 1. Determine Rotation Index
    rot_idx = action_idx // area
    
    # 2. Determine Spatial Coordinates
    remainder = action_idx % area
    y = remainder // width
    x = remainder % width
    
    return rot_idx, x, y
