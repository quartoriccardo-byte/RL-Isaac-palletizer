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
# Step 1: Parse arguments and launch app BEFORE any other imports
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
    
    # Add Isaac Lab launcher args
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    
    return parser.parse_args()


# Parse args first
args = parse_args()

# Launch Isaac Lab app (MUST be before other imports)
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# =============================================================================
# Step 2: Imports AFTER app launch
# =============================================================================

import torch
import gymnasium

# RSL-RL imports
from rsl_rl.runners import OnPolicyRunner

# Isaac Lab imports
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.wrappers.rsl_rl import RslRlVecEnvWrapper

# Project imports
from pallet_rl.envs.pallet_task import PalletTask, PalletTaskCfg
from pallet_rl.models.rsl_rl_wrapper import PalletizerActorCritic

# Load RSL-RL config
import yaml


# =============================================================================
# RSL-RL Configuration
# =============================================================================

def get_rsl_rl_cfg(args) -> dict:
    """
    Build RSL-RL configuration dictionary.
    
    Compatible with OnPolicyRunner expectations.
    """
    return {
        "seed": 42,
        
        "runner": {
            "policy_class_name": "ActorCritic",
            "algorithm_class_name": "PPO",
            "num_steps_per_env": 24,
            "max_iterations": args.max_iterations,
            "save_interval": 100,
            "experiment_name": args.experiment_name,
            "run_name": "run",
            "resume": args.resume,
            "load_run": -1,
            "checkpoint": -1 if args.checkpoint is None else args.checkpoint,
        },
        
        "policy": {
            "init_noise_std": 1.0,
            "actor_hidden_dims": [256, 256],
            "critic_hidden_dims": [256, 256],
            "activation": "elu",
        },
        
        "algorithm": {
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 3e-4,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
    }


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    """Main training entry point."""
    
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
