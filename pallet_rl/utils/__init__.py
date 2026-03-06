"""
pallet_rl.utils: Utility Functions and GPU Kernels

Heavy GPU imports (Warp, CUDA kernels) are deferred so that lightweight
utilities (quaternions, device_utils) can be imported without GPU runtime.
"""

try:
    from pallet_rl.utils.heightmap_rasterizer import WarpHeightmapGenerator
except ImportError:
    WarpHeightmapGenerator = None  # type: ignore[assignment,misc]

__all__ = ["WarpHeightmapGenerator"]
