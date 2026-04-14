"""
Action Adapter: Centralizes the semantic action space contract.

Supports two action modes:

1. **Factored discrete** (Stage D / legacy):
   Continuous Box([-1, 1] x 5) → factored 5-tuple (op, slot, x, y, rot).
   Used when buffer operations (STORE/RETRIEVE) are enabled.

2. **Joint place-only** (Stages A–C):
   Single categorical index → (x, y, rot) triple.
   The agent selects one placement from the full joint space.
   No buffer operations are available.
"""

from __future__ import annotations
from dataclasses import dataclass
import torch


# =========================================================================
# Factored Discrete (Stage D / legacy)
# =========================================================================

@dataclass
class DecodedAction:
    op_type: torch.Tensor   # 0=Place, 1=Store, 2=Retrieve
    slot_idx: torch.Tensor  # 0..9 (buffer slot)
    grid_x: torch.Tensor    # 0..X-1 (pallet X)
    grid_y: torch.Tensor    # 0..Y-1 (pallet Y)
    rot_idx: torch.Tensor   # 0=0°, 1=90°

def _to_discrete(a: torch.Tensor, k: int) -> torch.Tensor:
    """Map [-1, 1] → {0 .. k-1} uniformly."""
    return torch.floor(((a + 1.0) * 0.5) * k).long().clamp(0, k - 1)

def is_discrete_action_tensor(actions: torch.Tensor, action_dims: tuple[int, ...]) -> bool:
    """
    Check if a tensor should be treated as discrete factored indices.
    
    Criteria:
    1. Integer-type tensor -> True
    2. Float-type tensor where ALL values are integral AND within [0, k-1]
       for each corresponding dimension k -> True
    3. Otherwise -> False
    """
    if not torch.is_floating_point(actions):
        return True
    
    # Check if all values are integral
    is_integral = torch.all(actions == torch.floor(actions))
    if not is_integral:
        return False
    
    # Check if all values are in range
    for col, k in enumerate(action_dims):
        if torch.any((actions[:, col] < 0) | (actions[:, col] >= k)):
            return False
            
    return True

def discrete_to_center_normalized(actions: torch.Tensor, action_dims: tuple[int, ...]) -> torch.Tensor:
    """Map {0 .. k-1} → center-of-bin in [-1, 1]."""
    actions = actions.float()
    normalized = torch.zeros_like(actions)
    for col, k in enumerate(action_dims):
        normalized[:, col] = (actions[:, col] + 0.5) / k * 2.0 - 1.0
    return torch.clamp(normalized, -1.0, 1.0)

def decode_action_tensor(actions: torch.Tensor, action_dims: tuple[int, ...]) -> DecodedAction:
    """
    Centralized entry point for total action contract enforcement.
    
    1. Integer tensor -> discrete indices
    2. Float integral in-range tensor -> discrete indices
    3. Float in [-1, 1] -> normalized continuous
    4. Anything else -> raise ValueError
    """
    if is_discrete_action_tensor(actions, action_dims):
        # Treat as discrete
        return DecodedAction(
            op_type=actions[:, 0].long(),
            slot_idx=actions[:, 1].long(),
            grid_x=actions[:, 2].long(),
            grid_y=actions[:, 3].long(),
            rot_idx=actions[:, 4].long()
        )
    
    # Not discrete, check for normalized
    if torch.is_floating_point(actions) and torch.all((actions >= -1.0) & (actions <= 1.0)):
        return decode_normalized_action(actions, action_dims)
    
    raise ValueError(
        f"Ambiguous or invalid action tensor received.\n"
        f"Tensor: {actions}\n"
        f"Expected either discrete indices [0, k-1] or normalized floats in [-1, 1].\n"
        f"Mixed formats or values outside these ranges are strictly forbidden."
    )

def decode_normalized_action(action_norm: torch.Tensor, action_dims: tuple[int, ...]) -> DecodedAction:
    """
    Decode the continuous [-1, 1] actions from the policy into discrete semantics.
    
    Args:
        action_norm: Tensor of shape (N, 5) with values in [-1, 1].
        action_dims: Tuple of (op_dims, slot_dims, x_dims, y_dims, rot_dims).
        
    Returns:
        DecodedAction with integer tensors of shape (N,).
    """
    return DecodedAction(
        op_type=_to_discrete(action_norm[:, 0], action_dims[0]),
        slot_idx=_to_discrete(action_norm[:, 1], action_dims[1]),
        grid_x=_to_discrete(action_norm[:, 2], action_dims[2]),
        grid_y=_to_discrete(action_norm[:, 3], action_dims[3]),
        rot_idx=_to_discrete(action_norm[:, 4], action_dims[4])
    )


# =========================================================================
# Joint Place-Only (Stages A–C)
# =========================================================================

@dataclass
class DecodedPlaceAction:
    """Decoded joint spatial placement action for place-only stages."""
    grid_x: torch.Tensor    # 0..X-1
    grid_y: torch.Tensor    # 0..Y-1
    rot_idx: torch.Tensor   # 0..R-1


def decode_joint_place_action(
    action_idx: torch.Tensor,
    grid_x_dim: int,
    grid_y_dim: int,
    num_rotations: int,
) -> DecodedPlaceAction:
    """
    Decode a flat joint action index into (grid_x, grid_y, rot_idx).

    Encoding order: ``index = rot * (grid_y * grid_x) + y * grid_x + x``

    This is the canonical decoder for place-only curriculum stages.
    The joint index eliminates invalid factored combinations.

    Args:
        action_idx: (N,) tensor of joint action indices in [0, total_actions).
        grid_x_dim: Number of X grid cells.
        grid_y_dim: Number of Y grid cells.
        num_rotations: Number of rotation options (typically 2).

    Returns:
        DecodedPlaceAction with integer tensors of shape (N,).
    """
    action_idx = action_idx.long()
    spatial_size = grid_x_dim * grid_y_dim

    rot_idx = action_idx // spatial_size
    spatial = action_idx % spatial_size
    grid_y = spatial // grid_x_dim
    grid_x = spatial % grid_x_dim

    return DecodedPlaceAction(
        grid_x=grid_x.clamp(0, grid_x_dim - 1),
        grid_y=grid_y.clamp(0, grid_y_dim - 1),
        rot_idx=rot_idx.clamp(0, num_rotations - 1),
    )


def encode_joint_place_action(
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    rot_idx: torch.Tensor,
    grid_x_dim: int,
    grid_y_dim: int,
) -> torch.Tensor:
    """
    Encode (grid_x, grid_y, rot_idx) into a flat joint action index.

    Inverse of :func:`decode_joint_place_action`.

    Returns:
        (N,) tensor of flat action indices.
    """
    spatial_size = grid_x_dim * grid_y_dim
    return rot_idx.long() * spatial_size + grid_y.long() * grid_x_dim + grid_x.long()
