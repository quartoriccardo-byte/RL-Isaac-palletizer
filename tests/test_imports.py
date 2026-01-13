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
    from pallet_rl.models import actor_critic
    from pallet_rl.models import rsl_rl_wrapper
    from pallet_rl.models import policy_heads
    from pallet_rl.algo import utils
    print("✓ All submodules imported successfully")


def test_import_classes():
    """Test key class imports."""
    from pallet_rl.envs.pallet_task import PalletTask, PalletTaskCfg
    from pallet_rl.models.actor_critic import ActorCritic
    from pallet_rl.models.rsl_rl_wrapper import PalletizerActorCritic
    from pallet_rl.models.policy_heads import SpatialPolicyHead
    from pallet_rl.algo.utils import decode_action, load_config
    print("✓ All key classes imported successfully")


if __name__ == "__main__":
    test_import_package()
    test_import_submodules()
    test_import_classes()
    print("\n✅ All import tests passed!")
