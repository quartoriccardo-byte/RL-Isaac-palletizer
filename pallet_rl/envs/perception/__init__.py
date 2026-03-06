"""
Perception backends for heightmap generation.

Provides lazy imports for the two heightmap backends:
  - ``WarpBackend``       — analytical GPU rasterization (training)
  - ``DepthCameraBackend`` — depth camera sensor (sim-to-real)

Both conform to ``BaseHeightmapBackend`` and can be used interchangeably.

Usage::

    from pallet_rl.envs.perception import create_backend

    backend = create_backend("warp")     # or "depth_camera"
    heightmap = backend.generate(env)    # (N, H, W) tensor
"""

from __future__ import annotations

from pallet_rl.envs.perception.base import BaseHeightmapBackend


def create_backend(name: str, **kwargs) -> BaseHeightmapBackend:
    """
    Factory function to create a heightmap backend by name.

    Args:
        name: Backend name — ``"warp"`` or ``"depth_camera"``.
        **kwargs: Passed to the backend constructor.

    Returns:
        An instance of ``BaseHeightmapBackend``.
    """
    if name == "warp":
        from pallet_rl.envs.perception.warp_backend import WarpBackend
        return WarpBackend(**kwargs)
    elif name == "depth_camera":
        from pallet_rl.envs.perception.depth_camera_backend import DepthCameraBackend
        return DepthCameraBackend(**kwargs)
    else:
        raise ValueError(
            f"Unknown heightmap backend: {name!r}. "
            f"Available: 'warp', 'depth_camera'"
        )


__all__ = ["BaseHeightmapBackend", "create_backend"]
