"""
pallet_rl.models: Neural Network Models

PalletizerActorCritic requires rsl_rl (optional dependency).
Import is deferred so that lightweight usage / pytest works without it.
"""

try:
    from pallet_rl.models.rsl_rl_wrapper import PalletizerActorCritic
except ImportError:
    PalletizerActorCritic = None  # type: ignore[assignment,misc]

__all__ = ["PalletizerActorCritic"]
