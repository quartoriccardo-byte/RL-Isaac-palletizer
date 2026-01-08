
import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic
from pallet_rl.models.unet2d import UNet2D
from pallet_rl.models.encoder2d import Encoder2D

class PalletizerActorCritic(ActorCritic):
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, 
                 actor_hidden_dims=[256, 256, 256], 
                 critic_hidden_dims=[256, 256, 256], 
                 activation='elu', 
                 init_noise_std=1.0, 
                 **kwargs):
        super(ActorCritic, self).__init__()
        
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        
        # 1. Perception Specs (Matches config)
        self.image_dim = 64 * 64
        self.proprio_dim = num_actor_obs - self.image_dim
        
        # 2. Shared Backbone (Visual Encoder)
        # We reuse UNet or Encoder? User requested UNet2D + Encoder2D.
        # "Instantiate your custom UNet2D + Encoder2D... once as a shared visual feature extractor".
        # This implies a composite backbone?
        # Maybe UNet for dense features (segmentation-like) and Encoder for latent?
        # Or maybe Encoder extracts features FROM UNet?
        # Let's assume standard RL visual backbone: CNN -> Latent.
        # I'll use Encoder2D as the primary feature extractor for the latent vector.
        # And wrapping it inside a sequential if needed.
        
        self.feature_extractor = nn.Sequential(
            Encoder2D(in_channels=1, features=32), # Output: (B, 32, H, W)
            nn.AdaptiveAvgPool2d(1),              # Output: (B, 32, 1, 1)
            nn.Flatten(),                         # Output: (B, 32)
            nn.Linear(32, 128),
            nn.ELU()
        )
        
        feature_dim = 128 + self.proprio_dim
        
        # 3. Actor Head (MLP)
        actor_layers = []
        dims = [feature_dim] + actor_hidden_dims
        for i in range(len(dims)-1):
            actor_layers.append(nn.Linear(dims[i], dims[i+1]))
            actor_layers.append(nn.ELU())
        actor_layers.append(nn.Linear(dims[-1], num_actions))
        self.actor_head = nn.Sequential(*actor_layers)
        
        # 4. Critic Head (MLP)
        critic_layers = []
        dims = [feature_dim] + critic_hidden_dims
        for i in range(len(dims)-1):
            critic_layers.append(nn.Linear(dims[i], dims[i+1]))
            critic_layers.append(nn.ELU())
        critic_layers.append(nn.Linear(dims[-1], 1))
        self.critic_head = nn.Sequential(*critic_layers)
        
        # Noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        
        # Disable Init?
        # super().init_weights(...)

    def _process_obs(self, obs):
        """
        Internal method to slice and encode observations.
        Returns: concatenated features (visual + proprio)
        """
        # Slice
        # Assuming [image (N), proprio (M)]
        images = obs[:, :self.image_dim]
        proprio = obs[:, self.image_dim:]
        
        # Reshape Image
        images = images.view(-1, 1, 64, 64) # (B, 1, 64, 64)
        
        # Encode (Shared)
        visual_latent = self.feature_extractor(images)
        
        # Concatenate
        features = torch.cat([visual_latent, proprio], dim=1)
        return features

    def act(self, obs, **kwargs):
        features = self._process_obs(obs)
        
        # Update Distribution
        mean = self.actor_head(features)
        self.distribution = torch.distributions.Normal(mean, mean*0. + self.std)
        
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        # rsl-rl calls this after act() has set self.distribution usually?
        # OR it passes obs? 
        # RSL-RL PPO Algorithm:
        # actions_log_prob = self.actor_critic.get_actions_log_prob(actions)
        # It relies on self.distribution being set by act() or evaluate() called just before.
        # But wait, evaluate() sets distribution?
        # Let's ensure Robustness: If distribution is not set/fresh, this fails.
        # RSL-RL typical flow: act() -> buffer. update() -> evaluate blocks.
        # In update: evaluate(obs_batch, ...) is called.
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs):
        features = self._process_obs(obs)
        return self.actor_head(features)

    def evaluate(self, obs, **kwargs):
        """
        Called during PPO update to get value and log_prob of actions (implicitly via distribution update).
        Returns: value
        """
        features = self._process_obs(obs)
        
        # Critic Value
        value = self.critic_head(features)
        
        # Update distribution for log_prob calculation downstream
        mean = self.actor_head(features)
        self.distribution = torch.distributions.Normal(mean, mean*0. + self.std)
        
        return value
