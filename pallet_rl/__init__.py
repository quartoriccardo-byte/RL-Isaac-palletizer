"""
pallet_rl: Isaac Lab 4.0+ Palletizing RL Package

This package provides:
- PalletTask: DirectRLEnv for box palletizing
- PalletizerActorCritic: CNN-based actor-critic for RSL-RL
- WarpHeightmapGenerator: GPU heightmap rasterization

Gymnasium Registration:
    import pallet_rl
    env = gymnasium.make("Isaac-Palletizer-Direct-v0")
"""

from __future__ import annotations

import gymnasium

# Import main components
from pallet_rl.envs.pallet_task import PalletTask, PalletTaskCfg
from pallet_rl.models.rsl_rl_wrapper import PalletizerActorCritic
from pallet_rl.utils.heightmap_rasterizer import WarpHeightmapGenerator


# =============================================================================
# Gymnasium Registration
# =============================================================================

gymnasium.register(
    id="Isaac-Palletizer-Direct-v0",
    entry_point="pallet_rl.envs.pallet_task:PalletTask",
    kwargs={"cfg": PalletTaskCfg()},
    max_episode_steps=500,
)


# =============================================================================
# Package Info
# =============================================================================

__version__ = "2.0.0"
__author__ = "PalletRL Team"

__all__ = [
    "PalletTask",
    "PalletTaskCfg",
    "PalletizerActorCritic",
    "WarpHeightmapGenerator",
]
