import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic

class PalletizerActorCritic(ActorCritic):
    def __init__(self, num_obs, num_critic_obs, num_actions, **kwargs):
        # We ignore num_obs from arguments because we define a custom architecture structure
        # but we pass it to super init to satisfy the base class
        super().__init__(num_obs=num_obs, num_critic_obs=num_critic_obs, num_actions=num_actions, **kwargs)
        
        # 1. Visual Encoder (CNN) for Heightmap
        # Assumes input image is flattened in the first part of the observation
        # Input shape: [Batch, 1, 160, 240] -> Flattened size 38400
        self.visual_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(31360, 256),  # Adjusted for 160x240 input
            nn.ReLU()
        )
        
        # 2. Vector Encoder for Buffer State & Box Dimensions
        # Assumes the rest of the observation is the vector part
        # Input shape: [Batch, 53] (3 box dims + 50 buffer state flattened)
        self.vector_net = nn.Sequential(
            nn.Linear(53, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        
        # 3. Actor & Critic Heads
        # Input: 256 (Visual) + 64 (Vector) = 320 combined features
        self.actor_head = nn.Sequential(
            nn.Linear(320, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, num_actions)
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(320, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 1)
        )

    def _split_obs(self, obs):
        # Hardcoded split based on 160x240 image size
        # Image: 160*240 = 38400 pixels
        img = obs[:, :38400].view(-1, 1, 160, 240) 
        vec = obs[:, 38400:] 
        return img, vec

    def forward_actor(self, obs):
        img, vec = self._split_obs(obs)
        vis_feat = self.visual_net(img)
        vec_feat = self.vector_net(vec)
        fused = torch.cat([vis_feat, vec_feat], dim=1)
        return self.actor_head(fused)

    def forward_critic(self, obs):
        img, vec = self._split_obs(obs)
        vis_feat = self.visual_net(img)
        vec_feat = self.vector_net(vec)
        fused = torch.cat([vis_feat, vec_feat], dim=1)
        return self.critic_head(fused)
    
    @property
    def action_mean(self):
        return self.actor_head[-1].weight.data.fill_(0) # Placeholder
