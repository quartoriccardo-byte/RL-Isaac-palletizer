
import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic
from rsl_rl.utils import unpad_trajectories

from pallet_rl.models.unet2d import UNet2D
from pallet_rl.models.encoder2d import Encoder2D

class PalletizerActorCritic(ActorCritic):
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, 
                 actor_hidden_dims=[256, 256, 256], 
                 critic_hidden_dims=[256, 256, 256], 
                 activation='elu', 
                 init_noise_std=1.0, 
                 **kwargs):
        # We inherit but we will override init mostly
        super(ActorCritic, self).__init__()
        
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        
        # Custom Architecture
        # 1. Perception (Image) 64x64 = 4096
        self.image_dim = 64 * 64
        self.proprio_dim = num_actor_obs - self.image_dim
        
        # Encoders
        self.unet = UNet2D(in_ch=1, base=32)
        # Or Encoder?
        # Prompt says: "Initialize your custom UNet2D and Encoder2D".
        # Let's use Encoder2D for state? Or UNet for visual? 
        # Usually UNet is for dense output. Encoder for latent.
        # If we output actions, we probably want latent?
        # Let's use Encoder2D for the image part.
        self.visual_encoder = Encoder2D(in_channels=1, features=64)
        
        # MLP for proprioception + Visual Latent
        # Encoder2D input 1x64x64 -> ... Check Encoder2D.
        # Encoder2D: Conv -> ReLU -> BN ...
        # We need to know output size.
        # Modify Encoder or Add Adaptive Pooling.
        self.visual_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ELU()
        )
        
        # MLP Actor
        actor_input_dim = 128 + self.proprio_dim
        actor_layers = []
        layers = [actor_input_dim] + actor_hidden_dims
        for i in range(len(layers) - 1):
            actor_layers.append(nn.Linear(layers[i], layers[i + 1]))
            actor_layers.append(nn.ELU())
        actor_layers.append(nn.Linear(layers[-1], num_actions))
        self.actor = nn.Sequential(*actor_layers)
        
        # Value (Critic)
        # Symmetric
        critic_layers = []
        layers = [actor_input_dim] + critic_hidden_dims
        for i in range(len(layers) - 1):
            critic_layers.append(nn.Linear(layers[i], layers[i + 1]))
            critic_layers.append(nn.ELU())
        critic_layers.append(nn.Linear(layers[-1], 1))
        self.critic = nn.Sequential(*critic_layers)
        
        # Action Std
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        
        # Disable generic weights init?
        # super().init_weights(...)

    def forward(self, obs):
        # Slice obs
        # Assuming info comes first or last? Prompt: "first 64x64 values"
        images = obs[:, :self.image_dim]
        proprio = obs[:, self.image_dim:]
        
        # Reshape images
        # (B, 4096) -> (B, 1, 64, 64)
        images = images.view(-1, 1, 64, 64)
        
        # Vision
        feat_map = self.visual_encoder(images)
        visual_latent = self.visual_projector(feat_map)
        
        # Fusion
        combined = torch.cat([visual_latent, proprio], dim=1)
        
        return combined

    def act(self, obs, **kwargs):
        features = self.forward(obs)
        self.update_distribution(features)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs):
        features = self.forward(obs)
        return self.actor(features)

    def evaluate(self, obs, **kwargs):
        features = self.forward(obs)
        value = self.critic(features)
        return value

    def update_distribution(self, features):
        mean = self.actor(features)
        self.distribution = torch.distributions.Normal(mean, mean*0. + self.std)
