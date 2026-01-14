"""
Test that pallet_rl package can be imported.
"""

def test_import_package():
    """Test basic package import."""
    import pallet_rl
    print("✓ pallet_rl imported successfully")


def test_import_submodules():
    """Test submodule imports."""
    from pallet_rl.envs import pallet_task
    from pallet_rl.models import rsl_rl_wrapper
    from pallet_rl.algo import utils
    from pallet_rl.utils import heightmap_rasterizer, quaternions
    print("✓ Env, models, algo, and utils submodules imported successfully")


def test_import_classes():
    """Test key class imports."""
    from pallet_rl.envs.pallet_task import PalletTask, PalletTaskCfg
    from pallet_rl.models.rsl_rl_wrapper import PalletizerActorCritic
    from pallet_rl.algo.utils import decode_action, load_config
    from pallet_rl.utils.quaternions import wxyz_to_xyzw, xyzw_to_wxyz
    print("✓ All key classes imported successfully")


if __name__ == "__main__":
    test_import_package()
    test_import_submodules()
    test_import_classes()
    print("\n✅ All import tests passed!")
