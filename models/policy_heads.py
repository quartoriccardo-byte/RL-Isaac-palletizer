import torch
import torch.nn as nn

class SpatialPolicyHead(nn.Module):
    """
    Outputs probability map (H, W) for pick/place and discrete rotation logits.
    """
    def __init__(self, in_channels: int, num_rotations: int = 4):
        super().__init__()
        self.num_rotations = num_rotations
        # Final 1x1 conv to map features to action logits (Rotations, H, W)
        self.conv_out = nn.Conv2d(in_channels, num_rotations, kernel_size=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        logits = self.conv_out(x)  # Shape: (Batch, Rotations, H, W)
        
        # Flatten for PPO Categorical Distribution: (Batch, Actions)
        B = logits.shape[0]
        flat_logits = logits.view(B, -1)
        
        if mask is not None:
            # Mask must be broadcastable or same shape. 
            # Apply hard masking (-1e8) to invalid actions.
            flat_mask = mask.view(B, -1)
            flat_logits = flat_logits.masked_fill(~flat_mask.bool(), -1.0e8)
            
        return flat_logits
