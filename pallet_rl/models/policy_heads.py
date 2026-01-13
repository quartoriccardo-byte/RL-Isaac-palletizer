import torch
import torch.nn as nn

class SpatialPolicyHead(nn.Module):
    """
    Outputs flattened logits for spatial action selection with rotation.
    
    Output shape: (Batch, num_rotations * H * W)
    
    Action masking is applied by setting invalid action logits to -1e8.
    """
    def __init__(self, in_channels: int, num_rotations: int = 4):
        super().__init__()
        self.num_rotations = num_rotations
        # Final 1x1 conv to map features to action logits (Rotations, H, W)
        self.conv_out = nn.Conv2d(in_channels, num_rotations, kernel_size=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with optional action masking.
        
        Args:
            x: Input features, shape (Batch, C, H, W)
            mask: Optional action mask, shape (Batch, R*H*W) or (Batch, R, H, W)
                  True = valid action, False = invalid action
                  
        Returns:
            Flattened logits, shape (Batch, R*H*W)
        """
        logits = self.conv_out(x)  # Shape: (Batch, Rotations, H, W)
        
        # Flatten for PPO Categorical Distribution: (Batch, Actions)
        B, R, H, W = logits.shape
        flat_logits = logits.view(B, -1)  # (B, R*H*W)
        
        if mask is not None:
            flat_mask = mask.view(B, -1)
            
            # SHAPE CONTRACT: mask must match logits exactly
            assert flat_mask.shape == flat_logits.shape, (
                f"Mask shape {flat_mask.shape} != logits shape {flat_logits.shape}. "
                f"Expected mask with {R*H*W} elements per batch."
            )
            
            # Apply hard masking (-1e8) to invalid actions
            flat_logits = flat_logits.masked_fill(~flat_mask.bool(), -1.0e8)
            
        return flat_logits

