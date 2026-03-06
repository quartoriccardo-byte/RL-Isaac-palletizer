"""
Base class for heightmap generation backends.

All backends produce the same output shape ``(N, H, W)`` tensor in meters,
regardless of the underlying sensing modality (analytical Warp raster,
depth camera, or future LiDAR/point-cloud backends).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pallet_rl.envs.pallet_task import PalletTask


class BaseHeightmapBackend(ABC):
    """
    Abstract base class for heightmap generation.

    Subclasses must implement:
      - ``generate(env) -> torch.Tensor`` producing ``(N, H, W)`` heightmap
      - ``name`` property for logging/diagnostics

    The environment orchestrator calls ``backend.generate(env)`` once per
    observation step. The backend is free to use any data available on
    the environment instance (scene data, cameras, configs, etc.).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name (e.g. ``'warp'``, ``'depth_camera'``)."""
        ...

    @abstractmethod
    def generate(self, env: PalletTask) -> torch.Tensor:
        """
        Generate a heightmap for all environments.

        Args:
            env: The ``PalletTask`` instance (provides scene data, config, etc.)

        Returns:
            heightmap: ``(N, H, W)`` float tensor in meters above ground.
        """
        ...

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """
        Optional reset hook called on environment reset.

        Backends that cache state (e.g., depth decimation) should clear
        their caches for the specified environments.

        Args:
            env_ids: Indices of environments being reset, or ``None`` for all.
        """
        pass
