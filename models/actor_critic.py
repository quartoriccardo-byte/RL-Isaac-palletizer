
import torch
import torch.nn as nn
from pallet_rl.models.encoder2d import Encoder2D
from pallet_rl.models.policy_heads import SpatialPolicyHead

class ActorCritic(nn.Module):
    def __init__(self, in_channels=1, features=64, num_rotations=4):
        super().__init__()
        # 1. Perception (Shared) with Input Normalization
        self.encoder = nn.Sequential(
            nn.BatchNorm2d(in_channels), # Safety against non-normalized inputs
            Encoder2D(in_channels, features)
        )
        
        # 2. Actor (Spatial Policy)
        self.actor = SpatialPolicyHead(features, num_rotations)
        
        # 3. Critic (Value Function)
        self.critic = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(features, 1)
        )

    def forward(self, x, mask=None):
        # Returns: (logits, value)
        features = self.encoder(x)
        logits = self.actor(features, mask)
        value = self.critic(features)
        return logits, value

    def get_value(self, x):
        features = self.encoder(x)
        return self.critic(features)
