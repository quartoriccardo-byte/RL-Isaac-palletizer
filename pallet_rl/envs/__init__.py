"""
pallet_rl.envs: Isaac Lab Environment Module

Heavy Isaac Lab imports are deferred so that lightweight utility modules
can be imported and tested without the full Omniverse runtime.
"""

try:
    from pallet_rl.envs.pallet_task import PalletTask, PalletTaskCfg
except ImportError:
    PalletTask = None  # type: ignore[assignment,misc]
    PalletTaskCfg = None  # type: ignore[assignment,misc]

__all__ = ["PalletTask", "PalletTaskCfg"]
