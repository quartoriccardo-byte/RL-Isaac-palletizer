import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic as RslActorCritic

class ActorCritic(RslActorCritic):
    def __init__(self, num_obs, num_critic_obs, num_actions, **kwargs):
        # Initializing the base class
        # Note: We handle the input shapes manually in the forward pass
        super().__init__(num_obs=num_obs, num_critic_obs=num_critic_obs, num_actions=num_actions, **kwargs)
        
        # --------------------------------------------------------
        # 1. Visual Encoder (CNN) for Heightmap
        # Input: [Batch, 1, 160, 240] -> Flattened: 38400
        # --------------------------------------------------------
        self.visual_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(31360, 256),  # Projected to latent size 256
            nn.ReLU()
        )
        
        # --------------------------------------------------------
        # 2. Vector Encoder for Buffer State
        # Input: [Batch, Vector_Dim] (Remainder of obs)
        # --------------------------------------------------------
        self.vector_net = nn.Sequential(
            # We use a lazy linear layer to adapt to whatever vector size remains
            nn.LazyLinear(128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        
        # --------------------------------------------------------
        # 3. Policy & Value Heads (Fused Features)
        # Input: 256 (Visual) + 64 (Vector) = 320
        # --------------------------------------------------------
        self.actor_head = nn.Sequential(
            nn.Linear(320, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, num_actions) # MultiDiscrete Logits
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(320, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 1)
        )

        # Initialize lazy layers with a dummy pass
        with torch.no_grad():
            dummy_vec = torch.zeros(1, 53) # Approx buffer size
            self.vector_net(dummy_vec)

    def _split_obs(self, obs):
        """
        Splits the flattened observation tensor into Image and Vector components.
        Hardcoded for 160x240 resolution.
        """
        IMG_SIZE = 38400 # 160 * 240
        
        # Slicing
        img_flat = obs[:, :IMG_SIZE]
        vec_part = obs[:, IMG_SIZE:]
        
        # Reshape image for Conv2D: [Batch, Channels, Height, Width]
        img_tensor = img_flat.view(-1, 1, 160, 240)
        
        return img_tensor, vec_part

    def forward_actor(self, obs):
        img, vec = self._split_obs(obs)
        
        vis_feat = self.visual_net(img)
        vec_feat = self.vector_net(vec)
        
        # Fusion
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
        return self.actor_head[-1].weight.data.fill_(0)
