"""
pallet_rl: Isaac Lab 4.0+ Palletizing RL Package

This package provides:
- PalletTask: DirectRLEnv for box palletizing
- PalletizerActorCritic: CNN-based actor-critic for RSL-RL
- WarpHeightmapGenerator: GPU heightmap rasterization

Gymnasium Registration:
    import pallet_rl
    env = gymnasium.make("Isaac-Palletizer-Direct-v0")

Note: Heavy Isaac Lab / Warp imports are deferred to allow utility modules
(e.g., pallet_rl.utils.quaternions) to be tested without Omniverse runtime.
"""

from __future__ import annotations

# =============================================================================
# Lazy / Optional Imports for Isaac Lab and Warp
# =============================================================================
# These imports are intentionally deferred to avoid import-time side effects
# from heavy dependencies (Isaac Lab, Omniverse, NVIDIA Warp). This allows
# lightweight utility modules to be imported and tested in a standard Python
# environment (pytest) without requiring the full simulation runtime.

_ISAAC_AVAILABLE = False
_WARP_AVAILABLE = False

# Attempt to import Isaac Lab components
try:
    from pallet_rl.envs.pallet_task import PalletTask, PalletTaskCfg
    from pallet_rl.models.rsl_rl_wrapper import PalletizerActorCritic
    _ISAAC_AVAILABLE = True
except ImportError:
    # Isaac Lab / Omniverse not available - this is expected in test environments
    PalletTask = None
    PalletTaskCfg = None
    PalletizerActorCritic = None

# Attempt to import Warp components
try:
    from pallet_rl.utils.heightmap_rasterizer import WarpHeightmapGenerator
    _WARP_AVAILABLE = True
except ImportError:
    # Warp not available - this is expected in test environments
    WarpHeightmapGenerator = None


# =============================================================================
# Gymnasium Registration (only if Isaac Lab is available)
# =============================================================================
# Gymnasium registration is deferred until we confirm Isaac Lab is present.
# This prevents errors when importing pallet_rl for utility-only usage.

if _ISAAC_AVAILABLE and PalletTaskCfg is not None:
    import gymnasium
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
