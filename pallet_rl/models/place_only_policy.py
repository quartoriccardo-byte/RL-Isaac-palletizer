"""
PlaceOnlyActorCritic: Single-categorical CNN-based policy for place-only stages.

For curriculum stages A–C, the agent selects a single joint placement action
index from the space of all (x, y, rotation) combinations. This eliminates
the factored op/slot/x/y/rot interface and its invalid-combination problem.

Architecture (same visual/vector backbone as PalletizerActorCritic):
    Observation (38491-dim) → Split → [CNN Visual Encoder, MLP Vector Encoder] → Fusion → [Actor, Critic]

    Actor Head:  (N, 320) → (N, total_place_actions)  single Categorical
    Critic Head: (N, 320) → (N, 1) value estimate

Compatible with RSL-RL OnPolicyRunner and PPO.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Observation layout constants (shared with PalletizerActorCritic)
IMAGE_SHAPE = (160, 240)
VECTOR_DIM = 91
VECTOR_BOX_DIMS_SLICE = slice(60, 63)
VECTOR_MAX_STACK_HEIGHT_IDX = 67


class PlaceOnlyActorCritic(nn.Module):
    """
    CNN-based Actor-Critic for place-only curriculum stages.

    Outputs a single Categorical distribution over all joint (x, y, rot)
    placement actions.  No buffer operations (STORE/RETRIEVE) exist in
    this action space.
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [256, 128],
        critic_hidden_dims: list[int] = [256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        max_height: float = 2.0,
        # Place-only grid config (injected at construction time by train.py)
        grid_x: int = 8,
        grid_y: int = 12,
        num_rotations: int = 2,
        **kwargs,
    ):
        super().__init__()

        self.is_recurrent = False
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.max_height = max_height

        # Joint action space dimensions
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.num_rotations = num_rotations
        self.total_actions = grid_x * grid_y * num_rotations

        # Observation structure
        self.image_shape = IMAGE_SHAPE
        self.image_dim = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]  # 38400
        self.vector_dim = VECTOR_DIM  # 91

        # -----------------------------------------------------------------
        # Visual Encoder (CNN)
        # Input: (N, 1, 160, 240)  Output: (N, 256)
        # -----------------------------------------------------------------
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(64 * 20 * 30, 256),
            nn.ELU(),
        )

        # -----------------------------------------------------------------
        # Vector Encoder (MLP)
        # Input: (N, 91)  Output: (N, 64)
        # -----------------------------------------------------------------
        self.mlp = nn.Sequential(
            nn.Linear(self.vector_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )

        fusion_dim = 256 + 64  # 320

        # -----------------------------------------------------------------
        # Actor Head — single Categorical over joint actions
        # Input: (N, 320)  Output: (N, total_actions)
        # -----------------------------------------------------------------
        self.actor_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ELU(),
            nn.Linear(128, self.total_actions),
        )

        # -----------------------------------------------------------------
        # Critic Head
        # Input: (N, 320)  Output: (N, 1)
        # -----------------------------------------------------------------
        self.critic_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

        # Distribution storage
        self.distribution: torch.distributions.Categorical | None = None

        # Entropy tensor storage (RSL-RL expects sliceable attribute)
        self._entropy: torch.Tensor = torch.zeros(1)

        # RSL-RL normalization interface stubs
        self.actor_obs_normalization = False
        self.critic_obs_normalization = False
        self.actor_obs_normalizer = None
        self.critic_obs_normalizer = None
        self.obs_normalizer = None

    # =====================================================================
    # RSL-RL Interface
    # =====================================================================

    def update_normalization(self, obs: torch.Tensor) -> None:
        """No-op observation normalization (RSL-RL PPO callback)."""
        pass

    def reset(self, dones=None):
        """No-op reset (RSL-RL compatibility)."""
        pass

    # =====================================================================
    # Observation Processing
    # =====================================================================

    def _unwrap_obs(self, obs) -> torch.Tensor:
        """Unwrap TensorDict/dict/tuple to flat tensor."""
        if isinstance(obs, torch.Tensor):
            result = obs
        elif hasattr(obs, "get") and hasattr(obs, "__class__") and "TensorDict" in obs.__class__.__name__:
            if "policy" in obs.keys():
                result = obs["policy"]
            elif "obs" in obs.keys():
                result = obs["obs"]
            else:
                for key in obs.keys():
                    val = obs[key]
                    if isinstance(val, torch.Tensor):
                        result = val
                        break
                else:
                    raise ValueError(f"TensorDict: no tensor found. Keys: {list(obs.keys())}")
        elif isinstance(obs, dict):
            if "policy" in obs:
                result = obs["policy"]
            elif "obs" in obs:
                result = obs["obs"]
            else:
                for key, val in obs.items():
                    if isinstance(val, torch.Tensor):
                        result = val
                        break
                else:
                    raise ValueError(f"Dict: no tensor found. Keys: {list(obs.keys())}")
        elif isinstance(obs, tuple):
            result = obs[0] if isinstance(obs[0], torch.Tensor) else obs[0]
        else:
            raise TypeError(f"Unsupported obs type: {type(obs)}")

        if not isinstance(result, torch.Tensor):
            raise TypeError(f"Expected Tensor, got {type(result)}")
        if result.dim() == 1:
            result = result.unsqueeze(0)
        return result

    def _process_obs(self, obs) -> torch.Tensor:
        """Process observation into fused (N, 320) feature vector."""
        obs_tensor = self._unwrap_obs(obs)

        images = obs_tensor[:, : self.image_dim]
        images = images.view(-1, 1, self.image_shape[0], self.image_shape[1])

        vector = obs_tensor[:, self.image_dim : self.image_dim + self.vector_dim]

        vis_latent = self.cnn(images)
        vec_latent = self.mlp(vector)

        return torch.cat([vis_latent, vec_latent], dim=1)

    # =====================================================================
    # Action Mask (height + border based, from observations)
    # =====================================================================

    def _compute_action_mask_from_obs(self, obs) -> torch.Tensor:
        """
        Compute joint-action mask from current observation.

        This is a conservative policy-side fallback; the environment-side
        mask in ``get_action_mask_place_only`` is authoritative.
        """
        obs_tensor = self._unwrap_obs(obs)
        device = obs_tensor.device
        batch_size = obs_tensor.shape[0]

        mask = torch.ones(batch_size, self.total_actions, dtype=torch.bool, device=device)

        if obs_tensor.shape[1] < self.image_dim + self.vector_dim:
            return mask

        vector = obs_tensor[:, self.image_dim : self.image_dim + self.vector_dim]
        images = obs_tensor[:, : self.image_dim].view(-1, 1, *self.image_shape)
        heightmap = images[:, 0] * self.max_height  # (B, H, W) in meters

        current_dims = vector[:, VECTOR_BOX_DIMS_SLICE]  # (B, 3)
        box_h = current_dims[:, 2]
        max_stack_height = vector[:, VECTOR_MAX_STACK_HEIGHT_IDX] * 3.0

        pallet_x, pallet_y = 1.2, 0.8
        eps = 1e-6
        H, W = self.image_shape

        for r in range(self.num_rotations):
            if r == 0:
                dx, dy = current_dims[:, 0:1], current_dims[:, 1:2]
            else:
                dx, dy = current_dims[:, 1:2], current_dims[:, 0:1]

            step_x = pallet_x / self.grid_x
            step_y = pallet_y / self.grid_y
            cxs = torch.arange(self.grid_x, device=device).float() * step_x - pallet_x / 2 + step_x / 2
            cys = torch.arange(self.grid_y, device=device).float() * step_y - pallet_y / 2 + step_y / 2

            # Border mask: (B, gy, gx)
            x_valid = (cxs[None, :] - dx / 2 >= -pallet_x / 2 + eps) & \
                      (cxs[None, :] + dx / 2 <= pallet_x / 2 - eps)
            y_valid = (cys[None, :] - dy / 2 >= -pallet_y / 2 + eps) & \
                      (cys[None, :] + dy / 2 <= pallet_y / 2 - eps)
            xy_valid = y_valid.unsqueeze(-1) & x_valid.unsqueeze(1)  # (B, gy, gx)

            # Height mask (sampled at grid cell centers)
            for yi in range(self.grid_y):
                for xi in range(self.grid_x):
                    px_idx = min(int(xi / max(1, self.grid_x - 1) * (W - 1)), W - 1)
                    py_idx = min(int(yi / max(1, self.grid_y - 1) * (H - 1)), H - 1)
                    h_at_cell = heightmap[:, py_idx, px_idx]
                    predicted_top = h_at_cell + box_h
                    invalid = predicted_top > max_stack_height
                    xy_valid[:, yi, xi] &= ~invalid

            offset = r * (self.grid_x * self.grid_y)
            mask[:, offset : offset + self.grid_x * self.grid_y] &= xy_valid.reshape(batch_size, -1)

        # Safety: ensure at least one action valid
        all_masked = ~mask.any(dim=1)
        if all_masked.any():
            mask[all_masked, 0] = True

        return mask

    # =====================================================================
    # Act / Evaluate / Inference
    # =====================================================================

    def act(self, obs, action_mask: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        Sample a single joint placement action.

        Returns:
            actions: (N, 1) joint action indices.
        """
        features = self._process_obs(obs)
        logits = self.actor_head(features)  # (N, total_actions)

        auto_mask = self._compute_action_mask_from_obs(obs)
        if action_mask is not None:
            combined = action_mask & auto_mask
        else:
            combined = auto_mask
        logits = logits.masked_fill(~combined, -1e9)

        self.distribution = torch.distributions.Categorical(logits=logits)
        actions = self.distribution.sample()  # (N,)

        self._entropy = self.distribution.entropy()  # (N,)

        return actions.unsqueeze(-1)  # (N, 1) for RSL-RL action buffer

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of joint actions.

        Args:
            actions: (N, 1) or (N,) joint action indices.

        Returns:
            log_prob: (N,)
        """
        assert self.distribution is not None, "Call act() or evaluate() first."
        if actions.dim() == 2:
            actions = actions[:, 0]
        return self.distribution.log_prob(actions.long())

    def act_inference(self, obs) -> torch.Tensor:
        """Deterministic (argmax) action selection. Returns (N, 1)."""
        features = self._process_obs(obs)
        logits = self.actor_head(features)

        mask = self._compute_action_mask_from_obs(obs)
        logits = logits.masked_fill(~mask, -1e9)

        return torch.argmax(logits, dim=-1).unsqueeze(-1)

    def evaluate(self, obs, action_mask: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        Compute value estimate and populate distribution for log_prob.

        Returns:
            value: (N, 1) value estimates.
        """
        features = self._process_obs(obs)
        value = self.critic_head(features)

        logits = self.actor_head(features)
        auto_mask = self._compute_action_mask_from_obs(obs)
        if action_mask is not None:
            combined = action_mask & auto_mask
        else:
            combined = auto_mask
        logits = logits.masked_fill(~combined, -1e9)

        self.distribution = torch.distributions.Categorical(logits=logits)
        self._entropy = self.distribution.entropy()

        return value

    # =====================================================================
    # Properties (RSL-RL interface)
    # =====================================================================

    @property
    def entropy(self) -> torch.Tensor:
        """Precomputed entropy tensor (sliceable, as RSL-RL expects)."""
        return self._entropy

    @property
    def action_mean(self) -> torch.Tensor:
        """API stub — discrete policy has no action mean."""
        return torch.zeros(1, device=next(self.parameters()).device)

    @property
    def action_std(self) -> torch.Tensor:
        """API stub — discrete policy has no action std."""
        return torch.zeros(1, device=next(self.parameters()).device)

    @property
    def std(self) -> torch.Tensor:
        """API stub for RSL-RL PPO std access."""
        return torch.zeros(1, device=next(self.parameters()).device)
