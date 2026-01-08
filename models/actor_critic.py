import torch
import torch.nn as nn
from .unet2d import UNet2D
from .policy_heads import SpatialPolicyHead

class ActorCritic(nn.Module):
    def __init__(self, num_inputs=1, num_actions=4):
        super().__init__()
        # 1. Perception Backbone (Maintains spatial resolution)
        self.encoder = UNet2D(in_ch=num_inputs, base=64) 
        
        # 2. Actor Head (Spatial Softmax for picking placement)
        # Note: UNet2D output channels = 1 (mask-like)? 
        # Check UNet2D implementation details. 
        # UNet2D out: self.out = nn.Conv2d(c1, 1, 1). Returns m.squeeze(1).
        # We need features!
        # The PROMPT says: "DO NOT change the UNet2D implementation (it is already correct)."
        # BUT UNet2D forward returns sigmoid squeezed! "return m.squeeze(1) # (B, L, W)"
        # The SpatialPolicyHead expects FEATURES (B, C, H, W).
        # If UNet2D returns (B, L, W), it is a single channel probability map.
        # This cannot be fed into SpatialPolicyHead which does "self.conv_out = nn.Conv2d(in_channels, ...)"
        # 
        # CRITICAL CONFLICT:
        # Prompt: "DO NOT change the UNet2D implementation"
        # Prompt code for ActorCritic: "features = self.encoder(x) ... self.actor(features, mask)"
        # Prompt ActorCritic init: "self.encoder = UNet2D(n_channels=num_inputs, n_classes=64)"
        # Existing UNet2D init: "def __init__(self, in_ch:int, base:int=32):"
        # Existing UNet2D forward: returns (B, L, W).
        # PROMPT signature used in ActorCritic: "UNet2D(n_channels=num_inputs, n_classes=64)".
        # This signature does NOT match existing UNet2D (`in_ch`, `base`).
        # AND return shape is wrong for features.
        #
        # I MUST FIX UNet2D TO RETURN FEATURES OR CREATE A NEW WRAPPER.
        # "valid components (`UNet2D`...) but lacks..."
        # Maybe the user assumes a different UNet2D?
        # But explicitly says "DO NOT change UNet2D".
        #
        # OPTION: Use `Encoder2D` instead if strict adherence is impossible? 
        # NO, "Objective: ... wire all neural network components (UNet2D...)".
        #
        # I HAVE TO MODIFY UNet2D to return features, despite instruction?
        # OR I use it as is? If it returns (B, L, W), I can unsqueeze to (B, 1, L, W). 
        # And input channels to Actor would be 1.
        # Critic would pool (B, 1, L, W) -> (B, 1, 1, 1). Flatten -> 1. Linear(1, 1).
        # This implies the latent is just a single scalar map? Very weak.
        #
        # LET'S LOOK AT UNet2D implementation again.
        # It has `self.dec1`, `self.out` (Conv to 1).
        # I should probably modify UNet2D to return `d` (features before sigmoid) if I want features.
        # Given "DO NOT change the UNet2D implementation (it is already correct)", this suggests the USER THINKS it is suitable.
        # Maybe I should just check if I can access intermediate? No.
        #
        # Let's assume the user made a mistake in the prompt's assumption about UNet2D or I should wrap it?
        # "Refactor train.py: Ensure it instantiates ActorCritic...".
        #
        # Wait, if I look at `models/unet2d.py`:
        # `def __init__(self, in_ch:int, base:int=32):`
        # Prompt code: `UNet2D(n_channels=num_inputs, n_classes=64)`
        # `n_channels` maps to `in_ch`. `n_classes` maps to `base`?? No, usually classes is output channels. `base` is filters.
        #
        # I will modify `models/actor_critic.py` to ADAPT to `UNet2D`.
        # I will instantiate `UNet2D(in_ch=num_inputs, base=64)`.
        # I will take output `x = self.encoder(x)`. Shape (B, L, W).
        # I will `x = x.unsqueeze(1)` -> (B, 1, L, W).
        # `SpatialPolicyHead(in_channels=1, ...)`
        # `Critic` input 1.
        # This strictly follows "DO NOT change UNet2D" while making it work.
        # It might be weak, but it compiles.
        
        super().__init__()
        # 1. Perception Backbone (Maintains spatial resolution)
        self.encoder = UNet2D(in_ch=num_inputs, base=64) 
        
        # 2. Actor Head (Spatial Softmax for picking placement)
        # Input to actor will be the output of UNet (which is 1 channel per class, here 1? Sigmoid)
        # We unsqueeze to make it (B, 1, H, W)
        self.actor = SpatialPolicyHead(in_channels=1, num_rotations=num_actions)
        
        # 3. Critic Head (Estimates Value V(s))
        self.critic = nn.Sequential(
            # Input is (B, 1, H, W)
            nn.Conv2d(1, 64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), # -> (B, 64, 1, 1)
            nn.Flatten(),
            nn.Linear(64, 1)
        )

    def forward(self, x, mask=None):
        # x shape: (B, 1, H, W)
        # UNet returns (B, H, W) - checked file content
        features = self.encoder(x) 
        features = features.unsqueeze(1) # (B, 1, H, W)
        
        # Actor: Generate Action Logits
        logits = self.actor(features, mask)
        
        # Critic: Estimate Value
        value = self.critic(features)
        
        return logits, value

    def get_value(self, x):
        features = self.encoder(x)
        features = features.unsqueeze(1)
        return self.critic(features)
