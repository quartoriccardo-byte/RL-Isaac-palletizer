"""
Action Adapter: Centralizes the semantic action space contract.

This environment exposes a continuous Box([-1, 1] x 5) interface to the trainer
(e.g., RSL-RL PPO) for compatibility. However, the true semantic action space
is a factored discrete 5-tuple.

This module provides the rigorous mapping from the trainer's normalized float
tensor to the canonical DecodedAction representation used by all downstream
environment logic. No downstream module should consume the raw float actions.
"""

from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class DecodedAction:
    op_type: torch.Tensor   # 0=Place, 1=Store, 2=Retrieve
    slot_idx: torch.Tensor  # 0..9 (buffer slot)
    grid_x: torch.Tensor    # 0..15 (pallet X)
    grid_y: torch.Tensor    # 0..23 (pallet Y)
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
