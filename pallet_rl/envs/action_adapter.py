"""
Action Adapter: Centralizes the semantic action space contract.

This environment exposes a continuous Box([-1, 1] x 5) interface to the trainer
(e.g., RSL-RL PPO) for compatibility. However, the true semantic action space
is MultiDiscrete.

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
