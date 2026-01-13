
import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic
from pallet_rl.models.unet2d import UNet2D
from pallet_rl.models.encoder2d import Encoder2D


class PalletizerActorCritic(ActorCritic):
    """
    Custom ActorCritic for MultiDiscrete palletizing actions.
    
    Inherits from RSL-RL ActorCritic but overrides all methods to support:
    - CNN visual encoder for heightmap
    - MLP vector encoder for buffer state
    - MultiDiscrete action space (5 dimensions: op, slot, x, y, rot)
    """
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, 
                 actor_hidden_dims=[256, 256, 256], 
                 critic_hidden_dims=[256, 256, 256], 
                 activation='elu', 
                 init_noise_std=1.0, 
                 **kwargs):
        # FIXED: Was using super(ActorCritic, self).__init__() which skips
        # the RSL-RL ActorCritic.__init__() entirely (goes to nn.Module).
        # Since we override ALL methods and attributes, we intentionally
        # call nn.Module.__init__() directly - no base class state needed.
        nn.Module.__init__(self)
        
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions # This might be the sum of logits (55) or number of dims? 
        # RSL-RL wrapper passes num_actions from env.num_actions in train.py? 
        # In train.py: `num_actions = env.cfg.num_actions`.
        # In PalletTask, num_actions was set to 5 (num dims).
        # We need the ACTUAL counts for each dim.
        # Hardcoding per user spec for now as `action_dims`.
        self.action_dims = [3, 10, 16, 24, 2]
        self.total_logits = sum(self.action_dims)
        
        # 1. Perception Specs
        self.image_shape = (160, 240)
        self.image_dim = 160 * 240
        self.vector_dim = 53 # 3 (Box Dims) + 50 (Buffer)
        # Proprio (24) is excluded from Network Input per User Spec, but present in Obs.
        # We will strip it.
        
        # 2. Visual Head (CNN)
        # Input: (B, 1, 160, 240)
        # Output: 256
        # Simple 3-layer CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2), # (80, 120)
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # (40, 60)
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (20, 30)
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(64 * 20 * 30, 256),
            nn.ELU()
        )
        
        # 3. Vector Head (MLP)
        # Input: 53
        # Output: 64
        self.mlp = nn.Sequential(
            nn.Linear(self.vector_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU()
        )
        
        # 4. Fusion
        fusion_dim = 256 + 64 # 320
        
        # 5. Actor Head
        # Maps Fusion -> MultiDiscrete Logits (55)
        # User said "Linear layers mapping Fusion -> MultiDiscrete logits"
        self.actor_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ELU(),
            nn.Linear(128, self.total_logits)
        )
        
        # 6. Critic Head
        # Maps Fusion -> Scalar
        self.critic_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )
        
        # Init weights
        # self.apply(self.init_weights) # If needed

    def _process_obs(self, obs):
        # Flattened Obs: [Visual (38400) | Vector (53) | Proprio (24)]
        # Total 38477.
        
        # Slice Visual
        images = obs[:, :self.image_dim]
        images = images.view(-1, 1, 160, 240)
        
        # Slice Vector
        # 38400 : 38453
        vector = obs[:, self.image_dim : self.image_dim + self.vector_dim]
        
        # Proprio ignored
        
        vis_latent = self.cnn(images)
        vec_latent = self.mlp(vector)
        
        fusion = torch.cat([vis_latent, vec_latent], dim=1)
        return fusion

    def act(self, obs, **kwargs):
        features = self._process_obs(obs)
        logits = self.actor_head(features)
        
        # Split Logits & Sample
        actions = []
        start = 0
        log_probs = 0
        
        # To support RSL-RL which often expects distribution in self.distribution
        # We can create a list of distributions? Or one Independent Categorical?
        # Typically RSL-RL stores `self.distribution` for `get_actions_log_prob`.
        # We will create a comprehensive distribution object or list.
        # But RSL-RL Runner logic usually calls `get_actions_log_prob(actions)`.
        # So we need to store the logits or distributions.
        
        self.distributions = []
        action_list = []
        
        for dim in self.action_dims:
            end = start + dim
            l = logits[:, start:end]
            dist = torch.distributions.Categorical(logits=l)
            self.distributions.append(dist)
            
            a = dist.sample()
            action_list.append(a)
            start = end
            
        actions = torch.stack(action_list, dim=-1) # (B, 5)
        return actions

    def get_actions_log_prob(self, actions):
        # actions: (B, 5)
        log_prob = 0
        for i, dist in enumerate(self.distributions):
            a = actions[:, i]
            log_prob += dist.log_prob(a)
        
        return log_prob

    def act_inference(self, obs):
        features = self._process_obs(obs)
        logits = self.actor_head(features)
        
        # Deterministic: Argmax
        actions = []
        start = 0
        for dim in self.action_dims:
            end = start + dim
            l = logits[:, start:end]
            a = torch.argmax(l, dim=-1)
            actions.append(a)
            start = end
            
        return torch.stack(actions, dim=-1)

    def evaluate(self, obs, **kwargs):
        features = self._process_obs(obs)
        value = self.critic_head(features)
        
        # Re-populate distributions for downstream PPO
        logits = self.actor_head(features)
        self.distributions = []
        start = 0
        for dim in self.action_dims:
            end = start + dim
            l = logits[:, start:end]
            dist = torch.distributions.Categorical(logits=l)
            self.distributions.append(dist)
            start = end
            
        return value
