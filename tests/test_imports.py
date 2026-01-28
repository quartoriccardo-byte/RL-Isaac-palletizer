"""
Test that pallet_rl package can be imported.

Note: Some tests require the pxr module (Omniverse USD) and are skipped
when running in a standard Python environment without Isaac Lab runtime.
"""

import importlib.util
import pytest

# -----------------------------------------------------------------------------
# Runtime availability check for Omniverse/Isaac Lab
# -----------------------------------------------------------------------------
# Detect if pxr (Omniverse USD) is available. When running under isaaclab.sh
# or inside an Omniverse Kit environment, pxr is present. In a standard
# conda/venv Python environment, pxr is not available and Isaac-dependent
# tests should be skipped gracefully.
HAS_PXR = importlib.util.find_spec("pxr") is not None

# Skip reason used for Isaac-dependent tests
SKIP_REASON_PXR = (
    "Requires pxr (Omniverse USD) - run under Isaac Lab runtime or Kit environment"
)


def test_import_package():
    """Test basic package import."""
    import pallet_rl
    print("✓ pallet_rl imported successfully")


def test_import_lightweight_utils():
    """Test that lightweight utility modules can be imported without pxr."""
    from pallet_rl.utils import quaternions
    from pallet_rl.algo import utils
    from pallet_rl.utils.quaternions import wxyz_to_xyzw, xyzw_to_wxyz
    from pallet_rl.algo.utils import decode_action, load_config
    print("✓ Lightweight utility modules imported successfully")


@pytest.mark.skipif(not HAS_PXR, reason=SKIP_REASON_PXR)
def test_import_submodules():
    """Test submodule imports (requires pxr for Isaac Lab envs)."""
    from pallet_rl.envs import pallet_task
    from pallet_rl.models import rsl_rl_wrapper
    from pallet_rl.algo import utils
    from pallet_rl.utils import heightmap_rasterizer, quaternions
    print("✓ Env, models, algo, and utils submodules imported successfully")


@pytest.mark.skipif(not HAS_PXR, reason=SKIP_REASON_PXR)
def test_import_classes():
    """Test key class imports (requires pxr for Isaac Lab envs)."""
    from pallet_rl.envs.pallet_task import PalletTask, PalletTaskCfg
    from pallet_rl.models.rsl_rl_wrapper import PalletizerActorCritic
    from pallet_rl.algo.utils import decode_action, load_config
    from pallet_rl.utils.quaternions import wxyz_to_xyzw, xyzw_to_wxyz
    print("✓ All key classes imported successfully")


if __name__ == "__main__":
    test_import_package()
    test_import_lightweight_utils()
    if HAS_PXR:
        test_import_submodules()
        test_import_classes()
        print("\n✅ All import tests passed!")
    else:
        print("\n⚠ Isaac-dependent tests skipped (pxr not available)")
        print("✅ Lightweight import tests passed!")
