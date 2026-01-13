
import yaml
import torch

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def decode_action(action_idx: torch.Tensor, width: int, height: int, num_rotations: int):
    """
    Decodes a flat action index into (rot_idx, x, y) coordinates.
    
    Formula: index = rot * (H * W) + y * W + x
    
    Args:
        action_idx: Flat action index tensor or int
        width: Grid width (W dimension, columns)
        height: Grid height (H dimension, rows)
        num_rotations: Number of rotation options (unused but kept for API clarity)
        
    Returns:
        rot_idx, x, y: Decoded action components
        
    Examples:
        # For a 160x240 grid with 4 rotations:
        # Total actions = 4 * 160 * 240 = 153600
        >>> decode_action(0, width=240, height=160, num_rotations=4)
        (0, 0, 0)
        >>> decode_action(240, width=240, height=160, num_rotations=4)
        (0, 0, 1)  # y=1, x=0
    """
    # CRITICAL FIX: Was using width*width (square assumption)
    # Must use width*height for rectangular grids
    assert width > 0 and height > 0, f"Invalid grid dimensions: {width}x{height}"
    
    area = width * height  # FIXED: was 'width * width'
    
    # 1. Extract Rotation
    rot_idx = action_idx // area
    
    # 2. Extract Spatial Position
    spatial_idx = action_idx % area
    y = spatial_idx // width
    x = spatial_idx % width
    
    return rot_idx, x, y

