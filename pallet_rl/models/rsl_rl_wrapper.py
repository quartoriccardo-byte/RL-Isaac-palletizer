"""
PalletizerActorCritic: CNN-based Actor-Critic for RSL-RL

Custom policy for MultiDiscrete palletizing actions with:
- CNN visual encoder for heightmap
- MLP vector encoder for buffer state
- Fusion layer for combined features
- Separate actor/critic heads

Compatible with RSL-RL OnPolicyRunner and PPO.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# RSL-RL base class
from rsl_rl.modules import ActorCritic


class PalletizerActorCritic(ActorCritic):
    """
    CNN-based Actor-Critic for palletizing with MultiDiscrete actions.
    
    Architecture:
        Observation (38491-dim) → Split → [Visual Encoder, Vector Encoder] → Fusion → [Actor, Critic]
    
    Visual Encoder (CNN):
        Input: (N, 1, 160, 240) heightmap
        Output: (N, 256) latent
    
    Vector Encoder (MLP):
        Input: (N, 91) buffer + box dims + payload/mass + constraints + proprio
        Output: (N, 64) latent
    
    Fusion:
        Concat → (N, 320)
    
    Actor Head:
        (N, 320) → (N, 55) MultiDiscrete logits
    
    Critic Head:
        (N, 320) → (N, 1) value
    """
    
    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [256, 128],
        critic_hidden_dims: list[int] = [256, 128],
        activation: str = 'elu',
        init_noise_std: float = 1.0,
        **kwargs
    ):
        # Initialize nn.Module directly (we override everything)
        nn.Module.__init__(self)
        
        # Store config
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        
        # Action space definition
        self.action_dims = [3, 10, 16, 24, 2]  # Op, Slot, X, Y, Rot
        self.total_logits = sum(self.action_dims)  # 55
        
        # Observation structure
        # Updated for new constraints:
        # - Buffer features increased from 5 to 6 (added mass)
        # - Added payload_norm and current_box_mass_norm
        # - Added max_payload_norm and max_stack_height_norm (for future domain randomization)
        self.image_shape = (160, 240)
        self.image_dim = 160 * 240  # 38400
        # Buffer (60) + Box dims (3) + payload_norm (1) + mass_norm (1) + max_payload_norm (1) + max_stack_height_norm (1) + Proprio (24) = 91
        self.vector_dim = 91  # was 89
        
        # ---------------------------------------------------------------------
        # Visual Encoder (CNN)
        # Input: (N, 1, 160, 240)
        # Output: (N, 256)
        # ---------------------------------------------------------------------
        self.cnn = nn.Sequential(
            # Layer 1: (1, 160, 240) -> (16, 80, 120)
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            
            # Layer 2: (16, 80, 120) -> (32, 40, 60)
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            
            # Layer 3: (32, 40, 60) -> (64, 20, 30)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            
            # Flatten: 64 * 20 * 30 = 38400
            nn.Flatten(),
            
            # Project to latent
            nn.Linear(64 * 20 * 30, 256),
            nn.ELU()
        )
        
        # ---------------------------------------------------------------------
        # Vector Encoder (MLP)
        # Input: (N, 77) = Buffer (50) + Box dims (3) + Proprio (24)
        # Output: (N, 64)
        # ---------------------------------------------------------------------
        self.mlp = nn.Sequential(
            nn.Linear(self.vector_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU()
        )
        
        # Fusion dimension
        fusion_dim = 256 + 64  # 320
        
        # ---------------------------------------------------------------------
        # Actor Head
        # Input: (N, 320)
        # Output: (N, 55) - MultiDiscrete logits
        # ---------------------------------------------------------------------
        self.actor_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ELU(),
            nn.Linear(128, self.total_logits)
        )
        
        # ---------------------------------------------------------------------
        # Critic Head
        # Input: (N, 320)
        # Output: (N, 1)
        # ---------------------------------------------------------------------
        self.critic_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )
        
        # Distribution storage (set by act/evaluate)
        self.distributions: list = []
    
    def _process_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Process observation into fused feature vector.
        
        Args:
            obs: Flattened observation (N, 38491)
            
        Returns:
            fusion: Feature vector (N, 320)
        """
        # Split observation
        images = obs[:, :self.image_dim]
        images = images.view(-1, 1, 160, 240)
        
        # Vector = Buffer (50) + Box dims (3) + Proprio (24) = 77 dims
        vector = obs[:, self.image_dim:self.image_dim + self.vector_dim]
        
        # Encode
        vis_latent = self.cnn(images)
        vec_latent = self.mlp(vector)
        
        # Fuse
        fusion = torch.cat([vis_latent, vec_latent], dim=1)
        
        return fusion
    
    def act(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        Sample actions from policy.
        
        Args:
            obs: Observations (N, obs_dim)
            
        Returns:
            actions: Sampled actions (N, 5)
        """
        features = self._process_obs(obs)
        logits = self.actor_head(features)

        # Optional action masking: mask is shaped (N, total_logits)
        if action_mask is not None:
            assert action_mask.shape == logits.shape, (
                f"Action mask shape {action_mask.shape} must match logits {logits.shape}"
            )
            # Invalid actions get very negative logits so their prob ~ 0
            logits = logits.masked_fill(~action_mask, -1e9)
        
        # Split logits and create distributions
        self.distributions = []
        action_list = []
        start = 0
        
        for dim in self.action_dims:
            end = start + dim
            dim_logits = logits[:, start:end]
            
            dist = torch.distributions.Categorical(logits=dim_logits)
            self.distributions.append(dist)
            
            action_list.append(dist.sample())
            start = end
        
        actions = torch.stack(action_list, dim=-1)
        return actions
    
    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of actions.
        
        INVARIANT: self.distributions must be set by prior act() or evaluate().
        
        Args:
            actions: Actions (N, 5)
            
        Returns:
            log_prob: Log probabilities (N,)
        """
        assert len(self.distributions) == len(self.action_dims), \
            "distributions not set. Call act() or evaluate() first."
        
        log_prob = torch.zeros(actions.shape[0], device=actions.device)
        
        for i, dist in enumerate(self.distributions):
            log_prob = log_prob + dist.log_prob(actions[:, i])
        
        return log_prob
    
    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Deterministic action selection (argmax).
        
        Args:
            obs: Observations (N, obs_dim)
            
        Returns:
            actions: Deterministic actions (N, 5)
        """
        features = self._process_obs(obs)
        logits = self.actor_head(features)
        
        action_list = []
        start = 0
        
        for dim in self.action_dims:
            end = start + dim
            dim_logits = logits[:, start:end]
            action_list.append(torch.argmax(dim_logits, dim=-1))
            start = end
        
        return torch.stack(action_list, dim=-1)
    
    def evaluate(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        Compute value estimate and set distributions.
        
        Args:
            obs: Observations (N, obs_dim)
            
        Returns:
            value: Value estimates (N, 1)
        """
        features = self._process_obs(obs)
        value = self.critic_head(features)

        # Also populate distributions for get_actions_log_prob
        logits = self.actor_head(features)

        if action_mask is not None:
            assert action_mask.shape == logits.shape, (
                f"Action mask shape {action_mask.shape} must match logits {logits.shape}"
            )
            logits = logits.masked_fill(~action_mask, -1e9)
        self.distributions = []
        start = 0
        
        for dim in self.action_dims:
            end = start + dim
            dim_logits = logits[:, start:end]
            dist = torch.distributions.Categorical(logits=dim_logits)
            self.distributions.append(dist)
            start = end
        
        return value
    
    def entropy(self) -> torch.Tensor:
        """
        Compute entropy of action distributions.
        
        Used by RSL-RL PPO for entropy regularization.
        
        Returns:
            entropy: Sum of entropies (N,)
        """
        assert len(self.distributions) > 0, \
            "distributions not set. Call act() or evaluate() first."
        
        entropy = torch.zeros(
            self.distributions[0].probs.shape[0],
            device=self.distributions[0].probs.device
        )
        
        for dist in self.distributions:
            entropy = entropy + dist.entropy()
        
        return entropy
    
    @property
    def action_mean(self) -> torch.Tensor:
        """API compatibility - discrete policy has no action mean."""
        return torch.zeros(1, device=next(self.parameters()).device)
    
    @property
    def action_std(self) -> torch.Tensor:
        """API compatibility - discrete policy has no action std."""
        return torch.zeros(1, device=next(self.parameters()).device)
