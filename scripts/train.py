"""
Isaac Lab 4.0+ Training Script

High-performance training pipeline using:
- Isaac Lab AppLauncher
- RSL-RL OnPolicyRunner
- GPU-only data flow

Usage:
    python scripts/train.py --headless --num_envs 4096
"""

from __future__ import annotations

import argparse
import os
import sys

# =============================================================================
# Step 1: Parse arguments (no side effects at import time)
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train Palletizer with RSL-RL")
    
    # Simulation
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device")
    
    # Training
    parser.add_argument("--max_iterations", type=int, default=2000, help="Training iterations")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    
    # Logging
    parser.add_argument("--log_dir", type=str, default="runs/palletizer", help="Log directory")
    parser.add_argument("--experiment_name", type=str, default="palletizer_ppo", help="Experiment name")
    
    # Isaac Lab launcher args (defined locally to avoid importing AppLauncher before main())
    # These match the most common AppLauncher CLI args for compatibility.
    parser.add_argument("--livestream", type=int, default=0, help="Livestream mode (0=off, 1=native, 2=webrtc)")
    parser.add_argument("--enable_cameras", action="store_true", help="Enable camera sensors")
    parser.add_argument("--video", action="store_true", help="Record video")
    parser.add_argument("--video_length", type=int, default=200, help="Video length in steps")
    parser.add_argument("--video_interval", type=int, default=2000, help="Video recording interval")
    
    return parser.parse_args()


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    """Main training entry point."""
    # Parse arguments (inside main to avoid import-time side effects)
    args = parse_args()
    
    # Launch Isaac Lab app (MUST be before other imports that touch simulation)
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    
    # ==========================================================================
    # IMPORTS AFTER AppLauncher (required for Isaac Lab compatibility)
    # ==========================================================================
    import torch
    import yaml

    # RSL-RL imports with clear error message
    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError as e:
        raise ImportError(
            "Failed to import rsl_rl. Please install RSL-RL:\n"
            "  Option 1 (recommended): pip install git+https://github.com/leggedrobotics/rsl_rl.git@<commit>\n"
            "  Option 2: pip install -e /path/to/rsl_rl\n"
            "  Option 3: If using Isaac Lab environment, rsl_rl may already be available.\n"
            f"Original error: {e}"
        ) from e

    # Isaac Lab imports
    from isaaclab.envs import DirectRLEnvCfg
    try:
        # Older IsaacLab layout
        from isaaclab.envs.wrappers.rsl_rl import RslRlVecEnvWrapper
    except ModuleNotFoundError:
        # Newer layout (IsaacLab split package)
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


    # Project imports
    from pallet_rl.envs.pallet_task import PalletTask, PalletTaskCfg
    from pallet_rl.models.rsl_rl_wrapper import PalletizerActorCritic

    # ==========================================================================
    # RSL-RL Configuration
    # ==========================================================================

    def get_rsl_rl_cfg(args) -> dict:
        """
        Load and adapt the RSL-RL configuration dictionary.

        The base configuration is stored in `pallet_rl/configs/rsl_rl_config.yaml`.
        CLI arguments (iterations, resume, checkpoint, experiment_name) override
        the corresponding fields to keep a single source of truth for defaults.
        """
        # Path: scripts/../pallet_rl/configs/rsl_rl_config.yaml
        # Note: scripts/ is inside pallet_rl/ package root
        cfg_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "pallet_rl",
            "configs",
            "rsl_rl_config.yaml",
        )
        cfg_path = os.path.abspath(cfg_path)
        
        # Fallback: try sibling directory structure if primary path doesn't exist
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "configs",
                "rsl_rl_config.yaml",
            )
            cfg_path = os.path.abspath(cfg_path)

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Override a few fields from CLI for convenience
        runner_cfg = cfg.setdefault("runner", {})
        runner_cfg["max_iterations"] = args.max_iterations
        runner_cfg["experiment_name"] = args.experiment_name
        runner_cfg["resume"] = args.resume

        # RSL-RL encodes checkpoint selection via `load_run`/`checkpoint`.
        if args.checkpoint is not None:
            runner_cfg["checkpoint"] = args.checkpoint
        
        # =======================================================================
        # REQUIRED: obs_groups for RSL-RL OnPolicyRunner
        # Maps algorithm observation sets to env observation dict keys.
        # The env's _get_observations() returns {"policy": ..., "critic": ...}
        # =======================================================================
        runner_cfg.setdefault("obs_groups", {
            "policy": ["policy"],
            "critic": ["critic"],
        })

        return cfg

    # ==========================================================================
    # Training
    # ==========================================================================

    print(f"\n{'='*60}")
    print("Isaac Lab 4.0+ Palletizer Training")
    print(f"{'='*60}")
    print(f"Device: {args.device}")
    print(f"Environments: {args.num_envs}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Headless: {args.headless}")
    print(f"{'='*60}\n")
    
    # -------------------------------------------------------------------------
    # Step 1: Create environment configuration
    # -------------------------------------------------------------------------
    env_cfg = PalletTaskCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    
    # -------------------------------------------------------------------------
    # Step 2: Create environment
    # -------------------------------------------------------------------------
    print("Creating environment...")
    env = PalletTask(cfg=env_cfg, render_mode=None if args.headless else "rgb_array")
    
    # -------------------------------------------------------------------------
    # Step 3: Wrap for RSL-RL
    # -------------------------------------------------------------------------
    print("Wrapping environment for RSL-RL...")
    env = RslRlVecEnvWrapper(env)
    
    # -------------------------------------------------------------------------
    # Step 4: Inject custom policy class
    # -------------------------------------------------------------------------
    # RSL-RL uses module-level lookup for policy class
    # We monkey-patch to use our custom CNN-based policy
    import rsl_rl.modules
    rsl_rl.modules.ActorCritic = PalletizerActorCritic
    
    # -------------------------------------------------------------------------
    # Step 5: Create RSL-RL runner
    # -------------------------------------------------------------------------
    print("Initializing RSL-RL runner...")
    rsl_cfg = get_rsl_rl_cfg(args)
    
    runner = OnPolicyRunner(
        env=env,
        train_cfg=rsl_cfg,
        log_dir=args.log_dir,
        device=args.device
    )
    
    # -------------------------------------------------------------------------
    # Step 6: Load checkpoint if resuming
    # -------------------------------------------------------------------------
    if args.resume and args.checkpoint is not None:
        print(f"Resuming from checkpoint: {args.checkpoint}")
        runner.load(args.checkpoint)
    
    # -------------------------------------------------------------------------
    # Step 7: Train!
    # -------------------------------------------------------------------------
    print(f"\nStarting training for {args.max_iterations} iterations...\n")
    
    runner.learn(
        num_learning_iterations=args.max_iterations,
        init_at_random_ep_len=True
    )
    
    # -------------------------------------------------------------------------
    # Step 8: Cleanup
    # -------------------------------------------------------------------------
    print("\nTraining complete. Shutting down...")
    
    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()

