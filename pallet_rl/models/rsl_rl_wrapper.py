"""
PalletizerActorCritic: CNN-based Actor-Critic for RSL-RL

Custom policy for continuous Box actions decoded to factored discrete palletizing semantics with:
- CNN visual encoder for heightmap
- MLP vector encoder for buffer state
- Fusion layer for combined features
- Separate actor/critic heads

Compatible with RSL-RL OnPolicyRunner and PPO.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# RSL-RL base
cnn_out_dim: int = 256
hidden_dims: tuple[int, ...] = (128, 64)

# Observation layout constants
IMAGE_SHAPE = (160, 240)
VECTOR_DIM = 91
VECTOR_BOX_DIMS_SLICE = slice(60, 63)
VECTOR_MAX_STACK_HEIGHT_IDX = 67

"""
Vector observation layout:
- buffer (60)
- current_box_dims (3)
- payload_norm (1)
- current_box_mass_norm (1)
- max_payload_norm (1)
- max_stack_height_norm (1)
- proprio (24)
Total: 91
"""


def register_custom_policy():
    """
    Registers the custom PalletizerActorCritic with RSL-RL.

    Isaac Lab relies on RSL-RL for PPO generation. In this repo's 
    intended stack, the OnPolicyRunner resolves the policy class by 
    looking up `rsl_rl.modules.PolicyName`. The cleanest available 
    workaround to inject a custom architecture is to patch 
    this global namespace before the runner is instantiated.
    """
    import rsl_rl.modules
    rsl_rl.modules.ActorCritic = PalletizerActorCritic

from rsl_rl.modules import ActorCritic


class PalletizerActorCritic(ActorCritic):
    """
    CNN-based Actor-Critic mapping continuous Box outputs to factored discrete actions.
    
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
        (N, 320) → (N, 55) continuous action logits mapped to Factored Discrete
    
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
        max_height: float = 2.0,
        **kwargs
    ):
        # Initialize nn.Module directly (we override everything)
        nn.Module.__init__(self)
        
        # Store config
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.max_height = max_height
        
        # Action space definition
        self.action_dims = [3, 10, 16, 24, 2]  # Op, Slot, X, Y, Rot
        self.total_logits = sum(self.action_dims)  # 55
        
        # Observation structure
        # Updated for new constraints:
        # - Buffer features increased from 5 to 6 (added mass)
        # - Added payload_norm and current_box_mass_norm
        # - Added max_payload_norm and max_stack_height_norm (for future domain randomization)
        self.image_shape = IMAGE_SHAPE
        self.image_dim = self.image_shape[0] * self.image_shape[1]  # 38400
        # Buffer (60) + current_box_dims (3) + payload/mass/norms (4) + Proprio (24) = 91
        self.vector_dim = VECTOR_DIM
        
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
        
        # Vector Encoder (MLP)
        # Input: (N, 91) = Buffer (60) + Box dims (3) + payload/mass/constraints (4) + Proprio (24)
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
        # Output: (N, 55) - action logits
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
        
        # =====================================================================
        # Entropy tensor storage (RSL-RL compatibility)
        # =====================================================================
        # RSL-RL PPO expects policy.entropy to be a sliceable Tensor, not a
        # callable method. We store the computed entropy here and expose it
        # via a @property for subscripting (e.g. policy.entropy[:batch_size]).
        self._entropy: torch.Tensor = torch.zeros(1)
        
        # =====================================================================
        # RSL-RL ActorCritic normalization interface compatibility
        # =====================================================================
        # Since we skip the parent ActorCritic.__init__() (calling nn.Module.__init__
        # directly), we must manually provide the normalization attributes that
        # RSL-RL's PPO.update_normalization() expects.
        # Normalization is disabled for now; can be enabled by implementing
        # proper running-stats normalizers if needed.
        self.actor_obs_normalization = False
        self.critic_obs_normalization = False
        self.actor_obs_normalizer = None
        self.critic_obs_normalizer = None
        self.obs_normalizer = None
    
    def update_normalization(self, obs: torch.Tensor) -> None:
        """
        No-op observation normalization update.
        
        RSL-RL PPO calls this every step. Since we disabled normalization,
        we override with an empty implementation.
        
        Args:
            obs: Observation tensor (ignored)
        """
        pass
    
    def _unwrap_obs(self, obs) -> torch.Tensor:
        """
        Unwrap observations from TensorDict/dict to a flat torch.Tensor.
        
        RSL-RL OnPolicyRunner may pass observations as:
        - TensorDict with keys like "policy", "critic"
        - dict with similar structure
        - Raw torch.Tensor (legacy behavior)
        
        Args:
            obs: Observation in any of the above formats
            
        Returns:
            torch.Tensor: Flat observation tensor (N, obs_dim)
        """
        # Case 1: Already a torch.Tensor - use directly
        if isinstance(obs, torch.Tensor):
            result = obs
        # Case 2: TensorDict (from tensordict library)
        elif hasattr(obs, 'get') and hasattr(obs, '__class__') and 'TensorDict' in obs.__class__.__name__:
            # Try known keys in priority order
            if "policy" in obs.keys():
                result = obs["policy"]
            elif "obs" in obs.keys():
                result = obs["obs"]
            else:
                # Fallback: get first tensor value
                for key in obs.keys():
                    val = obs[key]
                    if isinstance(val, torch.Tensor):
                        result = val
                        break
                else:
                    raise ValueError(f"TensorDict has no extractable tensor. Keys: {list(obs.keys())}")
        # Case 3: Regular dict
        elif isinstance(obs, dict):
            if "policy" in obs:
                result = obs["policy"]
            elif "obs" in obs:
                result = obs["obs"]
            else:
                # Fallback: get first tensor value
                for key, val in obs.items():
                    if isinstance(val, torch.Tensor):
                        result = val
                        break
                else:
                    raise ValueError(f"Dict has no extractable tensor. Keys: {list(obs.keys())}")
        else:
            raise TypeError(f"Unsupported observation type: {type(obs)}")
        
        # Ensure result is a tensor
        assert isinstance(result, torch.Tensor), f"Expected torch.Tensor, got {type(result)}"
        
        # Handle 1D case (single env) by adding batch dimension
        if result.dim() == 1:
            result = result.unsqueeze(0)
        
        assert result.dim() == 2, f"Expected 2D tensor (N, obs_dim), got shape {result.shape}"
        
        return result
    
    def _process_obs(self, obs) -> torch.Tensor:
        """
        Process observation into fused feature vector.
        
        Args:
            obs: Observation (N, 38491) as Tensor, TensorDict, or dict
            
        Returns:
            fusion: Feature vector (N, 320)
        """
        # Unwrap to tensor if needed
        obs_tensor = self._unwrap_obs(obs)
        
        # Split observation
        images = obs_tensor[:, :self.image_dim]
        images = images.view(-1, 1, self.image_shape[0], self.image_shape[1])
        
        # Vector = Buffer (60) + Box dims (3) + payload/mass/constraints (4) + Proprio (24) = 91 dims
        vector = obs_tensor[:, self.image_dim:self.image_dim + self.vector_dim]
        
        # Encode
        vis_latent = self.cnn(images)
        vec_latent = self.mlp(vector)
        
        # Fuse
        fusion = torch.cat([vis_latent, vec_latent], dim=1)
        
        return fusion
    
    def _compute_action_mask_from_obs(self, obs) -> torch.Tensor:
        """
        Compute a height-based action mask directly from observations.

        This provides an auxiliary height-based filter directly from observations.
        NOTE: This wrapper-side mask depends strictly on the current observation 
        vector layout. The environment-side validity logic in `PlacementController` 
        is the authoritative source of truth for physical constraints.
        This policy-side mask is a conservative fallback.
        """
        obs_tensor = self._unwrap_obs(obs)
        device = obs_tensor.device
        batch_size = obs_tensor.shape[0]

        mask = torch.ones(batch_size, self.total_logits, dtype=torch.bool, device=device)

        # Guard against unexpected observation layouts
        if obs_tensor.shape[1] < self.image_dim + self.vector_dim:
            return mask

        # Extract vector part and verify dimension
        vector = obs_tensor[:, self.image_dim : self.image_dim + self.vector_dim]
        assert vector.shape[1] == VECTOR_DIM, f"Unexpected vector dim: {vector.shape[1]} != {VECTOR_DIM}"

        # Heightmap (normalized to [0,1]) → meters
        images = obs_tensor[:, :self.image_dim].view(-1, 1, self.image_shape[0], self.image_shape[1])
        heightmap_norm = images[:, 0]
        # max_height is stored on policy instance
        heightmap = heightmap_norm * self.max_height

        # Extraction logic depends strictly on the observation vector layout.
        current_dims = vector[:, VECTOR_BOX_DIMS_SLICE]
        box_h = current_dims[:, 2]

        num_x = self.action_dims[2]
        num_y = self.action_dims[3]
        image_h, image_w = self.image_shape
        
        grid_xs = torch.arange(num_x, device=device)
        grid_ys = torch.arange(num_y, device=device)

        # --- 1. Conservative Border Mask ---
        # Constants from canonical environment configuration
        pallet_x, pallet_y = 1.2, 0.8
        eps = 1e-6
        
        step_x = pallet_x / num_x
        step_y = pallet_y / num_y
        cxs = grid_xs.float() * step_x - pallet_x/2 + step_x/2
        cys = grid_ys.float() * step_y - pallet_y/2 + step_y/2
        
        # Effective XY for both rotations (swap X/Y for rot 1)
        # Current box dims from observation: current_dims (B, 3)
        box_dx0 = current_dims[:, 0:1] # (B, 1)
        box_dy0 = current_dims[:, 1:2] # (B, 1)
        
        # X Border Mask
        x_valid_rot0 = (cxs[None, :] - box_dx0/2 >= -pallet_x/2 + eps) & (cxs[None, :] + box_dx0/2 <= pallet_x/2 - eps)
        x_valid_rot1 = (cxs[None, :] - box_dy0/2 >= -pallet_x/2 + eps) & (cxs[None, :] + box_dy0/2 <= pallet_x/2 - eps)
        x_border_valid = x_valid_rot0 | x_valid_rot1
        mask[:, x_start : x_start + num_x] &= x_border_valid
        
        # Y Border Mask
        y_valid_rot0 = (cys[None, :] - box_dy0/2 >= -pallet_y/2 + eps) & (cys[None, :] + box_dy0/2 <= pallet_y/2 - eps)
        y_valid_rot1 = (cys[None, :] - box_dx0/2 >= -pallet_y/2 + eps) & (cys[None, :] + box_dx0/2 <= pallet_y/2 - eps)
        y_border_valid = y_valid_rot0 | y_valid_rot1
        mask[:, y_start : y_start + num_y] &= y_border_valid

        # --- 2. Height-based Fallback Mask ---
        pixel_xs = (grid_xs.float() / max(1, num_x - 1) * (image_w - 1)).long().clamp(0, image_w - 1)
        pixel_ys = (grid_ys.float() / max(1, num_y - 1) * (image_h - 1)).long().clamp(0, image_h - 1)

        # all_heights: (B, num_y, num_x)
        all_heights = heightmap[:, pixel_ys[:, None], pixel_xs[None, :]]
        predicted_tops = all_heights + box_h[:, None, None]

        # max_stack_height is derived from observation (normalized by 3.0m)
        max_stack_height = vector[:, VECTOR_MAX_STACK_HEIGHT_IDX] * 3.0
        grid_invalid = predicted_tops > max_stack_height[:, None, None]

        all_y_invalid_at_x = grid_invalid.all(dim=1)  # (B, num_x)
        all_x_invalid_at_y = grid_invalid.all(dim=2)  # (B, num_y)

        # Mask out logits corresponding to grid X/Y that are invalid for
        # all Y/X respectively. 
        # WARNING: This fallback mask is non-authoritative and depends 
        # strictly on the observation vector layout defined in PalletTask.
        # It is used as a safety prior for the actor.
        mask[:, x_start : x_start + num_x] &= ~all_y_invalid_at_x
        mask[:, y_start : y_start + num_y] &= ~all_x_invalid_at_y

        return mask
    
    def act(self, obs, action_mask: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        Sample actions from policy.
        
        Args:
            obs: Observations (N, obs_dim)
            
        Returns:
            actions: Sampled actions (N, 5)
        """
        features = self._process_obs(obs)
        logits = self.actor_head(features)
        
        # Height-based mask derived from observations (train + eval)
        auto_mask = self._compute_action_mask_from_obs(obs)

        # Optional external mask is AND-ed with the auto mask.
        if action_mask is not None:
            assert action_mask.shape == logits.shape, (
                f"Action mask shape {action_mask.shape} must match logits {logits.shape}"
            )
            combined_mask = action_mask & auto_mask
        else:
            combined_mask = auto_mask

        logits = logits.masked_fill(~combined_mask, -1e9)
        
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
        
        # Compute and store entropy tensor (RSL-RL expects this as attribute)
        self._entropy = self._compute_entropy()
        
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
    
    def act_inference(self, obs) -> torch.Tensor:
        """
        Deterministic action selection (argmax).
        
        Args:
            obs: Observations (N, obs_dim)
            
        Returns:
            actions: Deterministic actions (N, 5)
        """
        features = self._process_obs(obs)
        logits = self.actor_head(features)
        
        # Deterministic inference must respect the same validity mask as
        # training to keep evaluation semantics honest.
        mask = self._compute_action_mask_from_obs(obs)
        logits = logits.masked_fill(~mask, -1e9)
        
        action_list = []
        start = 0
        
        for dim in self.action_dims:
            end = start + dim
            dim_logits = logits[:, start:end]
            action_list.append(torch.argmax(dim_logits, dim=-1))
            start = end
        
        return torch.stack(action_list, dim=-1)
    
    def evaluate(self, obs, action_mask: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
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

        auto_mask = self._compute_action_mask_from_obs(obs)
        if action_mask is not None:
            assert action_mask.shape == logits.shape, (
                f"Action mask shape {action_mask.shape} must match logits {logits.shape}"
            )
            combined_mask = action_mask & auto_mask
        else:
            combined_mask = auto_mask
        logits = logits.masked_fill(~combined_mask, -1e9)
        self.distributions = []
        start = 0
        
        for dim in self.action_dims:
            end = start + dim
            dim_logits = logits[:, start:end]
            dist = torch.distributions.Categorical(logits=dim_logits)
            self.distributions.append(dist)
            start = end
        
        # Compute and store entropy tensor (RSL-RL expects this as attribute)
        self._entropy = self._compute_entropy()
        
        return value
    
    def _compute_entropy(self) -> torch.Tensor:
        """
        Compute entropy of action distributions (internal helper).
        
        Returns:
            entropy: Sum of entropies across action dims (N,)
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
    def entropy(self) -> torch.Tensor:
        """
        Entropy tensor (RSL-RL PPO interface).
        
        RSL-RL expects policy.entropy to be a Tensor attribute that can be
        sliced (e.g. policy.entropy[:batch_size]). This property returns
        the precomputed entropy stored by act() or evaluate().
        
        Returns:
            entropy: Sum of entropies (N,)
        """
        return self._entropy
    
    @property
    def action_mean(self) -> torch.Tensor:
        """API compatibility - discrete policy has no action mean."""
        return torch.zeros(1, device=next(self.parameters()).device)
    
    @property
    def action_std(self) -> torch.Tensor:
        """API compatibility - discrete policy has no action std."""
        return torch.zeros(1, device=next(self.parameters()).device)
