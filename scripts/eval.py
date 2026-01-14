
"""
Canonical evaluation script for the PalletTask + RSL-RL pipeline.

This script mirrors the training setup in `scripts/train.py` but:
- Loads a trained checkpoint.
- Runs a limited number of evaluation episodes.

NOTE: This file is not executed in this environment. It is structurally
correct by design and should be validated on a machine with Isaac Lab.
"""

from __future__ import annotations

import argparse
import os

import torch

from isaaclab.app import AppLauncher
from isaaclab.envs.wrappers.rsl_rl import RslRlVecEnvWrapper

from rsl_rl.runners import OnPolicyRunner

from pallet_rl.envs.pallet_task import PalletTask, PalletTaskCfg
from pallet_rl.models.rsl_rl_wrapper import PalletizerActorCritic


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Palletizer policy (Isaac Lab + RSL-RL)")

    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel environments")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to RSL-RL checkpoint (.pt)")
    parser.add_argument("--max_episodes", type=int, default=10, help="Max evaluation episodes per env")
    parser.add_argument("--log_dir", type=str, default="runs/eval", help="Eval log directory")

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    # Launch Isaac Lab app before other imports that touch simulation
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        device = torch.device(args.device)

        # Environment configuration
        env_cfg = PalletTaskCfg()
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = args.device

        render_mode = None if args.headless else "rgb_array"
        env = PalletTask(cfg=env_cfg, render_mode=render_mode)
        env = RslRlVecEnvWrapper(env)

        # Build minimal RSL-RL config for evaluation
        eval_cfg = {
            "seed": 42,
            "runner": {
                "policy_class_name": "ActorCritic",
                "algorithm_class_name": "PPO",
                "num_steps_per_env": 16,
                "max_iterations": 1,
                "save_interval": 0,
                "experiment_name": "palletizer_eval",
                "run_name": "eval",
                "resume": True,
                "load_run": -1,
                "checkpoint": args.checkpoint,
            },
            "policy": {
                "init_noise_std": 0.0,
                "actor_hidden_dims": [256, 128],
                "critic_hidden_dims": [256, 128],
                "activation": "elu",
            },
            "algorithm": {
                "value_loss_coef": 1.0,
                "use_clipped_value_loss": True,
                "clip_param": 0.2,
                "entropy_coef": 0.0,
                "num_learning_epochs": 1,
                "num_mini_batches": 1,
                "learning_rate": 3e-4,
                "schedule": "fixed",
                "gamma": 0.99,
                "lam": 0.95,
                "desired_kl": 0.01,
                "max_grad_norm": 1.0,
            },
        }

        # Inject our custom policy class
        import rsl_rl.modules

        rsl_rl.modules.ActorCritic = PalletizerActorCritic

        runner = OnPolicyRunner(env=env, train_cfg=eval_cfg, log_dir=args.log_dir, device=str(device))

        # Load checkpoint into runner/policy
        runner.load(args.checkpoint)

        # Simple evaluation loop using deterministic actions
        obs = runner.env.reset()
        episode_counts = torch.zeros(args.num_envs, dtype=torch.long, device=device)

        print("Starting evaluation rollouts...")

        while int(episode_counts.min().item()) < args.max_episodes:
            with torch.no_grad():
                actions = runner.alg.actor_critic.act_inference(obs["policy"])

            obs, rewards, dones, infos = runner.env.step(actions)

            # Count completed episodes
            if "time_outs" in infos:
                done_flags = dones | infos["time_outs"]
            else:
                done_flags = dones
            episode_counts += done_flags.to(device=device, dtype=torch.long)

        print("Evaluation complete.")

    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()

