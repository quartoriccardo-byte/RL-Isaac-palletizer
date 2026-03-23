"""
Evaluation script for Palletizer policies.

Usage:
    python scripts/eval.py --checkpoint /path/to/model.pt --num_envs 16

    # Multi-GPU workstation with Kit/Carb overrides:
    CUDA_VISIBLE_DEVICES=2 ~/isaac-sim/python.sh scripts/eval.py \
      --enable_cameras --num_envs 1 --device cuda:0 \
      --checkpoint /path/to/model_50.pt --max_episodes 3 \
      --/physics/cudaDevice=0 --/renderer/activeGpu=0 \
      --/renderer/multiGpu/enabled=false --/ngx/enabled=false
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import traceback
import torch
from pallet_rl.utils.device_utils import pick_supported_cuda_device


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    """Parse command-line arguments, allowing unknown Kit/Carb settings.

    Returns:
        tuple: (args, unknown) where args is the Namespace of known args,
               and unknown is a list of unrecognized arguments (e.g., --/rtx/...).
    """
    parser = argparse.ArgumentParser(description="Evaluate Palletizer policy (Isaac Lab + RSL-RL)")

    # Simulation
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel environments")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")

    # Evaluation
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to RSL-RL checkpoint (.pt)")
    parser.add_argument("--max_episodes", type=int, default=10, help="Max evaluation episodes per env")
    parser.add_argument("--log_dir", type=str, default="runs/eval", help="Eval log directory")

    # Isaac Lab launcher args (must be defined before AppLauncher init)
    parser.add_argument("--livestream", type=int, default=0, help="Livestream mode (0=off, 1=native, 2=webrtc)")
    parser.add_argument("--enable_cameras", action="store_true", default=False, help="Enable camera sensors")

    # Use parse_known_args to accept unknown Kit/Carb settings (--/path=value)
    args, unknown = parser.parse_known_args()
    return args, unknown


# =============================================================================
# Local helpers for carb settings
# =============================================================================

def parse_carb_setting(value_str: str):
    """Parse a carb setting value string to appropriate Python type.

    Converts:
        - 'true'/'false' -> bool
        - integer strings -> int
        - float strings -> float
        - otherwise -> str
    """
    if value_str.lower() == 'true':
        return True
    if value_str.lower() == 'false':
        return False

    try:
        return int(value_str)
    except ValueError:
        pass

    try:
        return float(value_str)
    except ValueError:
        pass

    return value_str


def apply_carb_settings(unknown_args: list) -> None:
    """Apply Kit/Carb settings from unknown arguments.

    Parses arguments of the form --/some/path=value and applies them
    via carb.settings.get_settings().set().

    Args:
        unknown_args: List of unrecognized CLI arguments.
    """
    import carb
    settings = carb.settings.get_settings()

    pattern = re.compile(r'^--(/[^=]+)=(.*)$')

    for arg in unknown_args:
        match = pattern.match(arg)
        if match:
            path = match.group(1)
            value_str = match.group(2)
            value = parse_carb_setting(value_str)
            settings.set(path, value)
            print(f"[EVAL][INFO] Applied carb setting: {path} = {value}")
        elif arg.startswith('--/'):
            print(f"[EVAL] [WARN] Malformed Kit/Carb arg (missing '='?): {arg}")


def unwrap_obs(obs):
    """Handle tuple returns from wrapped environments."""
    if isinstance(obs, tuple):
        return obs[0]
    return obs


# =============================================================================
# Main evaluation
# =============================================================================

def main():
    """Main evaluation entry point."""
    print("[EVAL] Entered main — starting evaluation script", flush=True)

    # ------------------------------------------------------------------
    # Step 1: Parse arguments
    # ------------------------------------------------------------------
    args, unknown = parse_args()

    # Selective GPU selection (RTX 6000 vs 1080 Ti)
    if args.device == "cuda":
        _, forced_device = pick_supported_cuda_device()
        args.device = forced_device
        print(f"[EVAL][INFO] Auto-selected supported GPU: {args.device}")
    else:
        print(f"[EVAL][INFO] Using user-specified device: {args.device}")

    # ------------------------------------------------------------------
    # Step 2: Validate checkpoint path early
    # ------------------------------------------------------------------
    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"  (original arg: {args.checkpoint})\n"
            "Please provide a valid --checkpoint path."
        )
    args.checkpoint = checkpoint_path
    print(f"[EVAL][INFO] Resolved checkpoint: {checkpoint_path}")

    # ------------------------------------------------------------------
    # Step 3: Inject Kit startup args into sys.argv BEFORE AppLauncher
    # ------------------------------------------------------------------
    # NGX/DLSS settings must be in sys.argv BEFORE Kit initializes.
    # AppLauncher/SimulationApp reads sys.argv during construction.

    # Collect user-provided Kit args from CLI
    user_kit_args = [arg for arg in unknown if arg.startswith('--/')]
    user_kit_paths = {arg.split('=')[0] for arg in user_kit_args}

    # Default Kit args for NGX/DLSS/RTX disabling.
    # Always inject to keep eval simple and safe; user args win via path dedup.
    # NOTE: --/rtx/post/dlss/enabled is intentionally EXCLUDED (known-problematic).
    defaults_map = {
        '--/ngx/enabled':                       '--/ngx/enabled=false',
        '--/rtx/post/dlss/execMode':            '--/rtx/post/dlss/execMode=0',
        '--/rtx-transient/dlssg/enabled':       '--/rtx-transient/dlssg/enabled=false',
        '--/rtx-transient/dldenoiser/enabled':  '--/rtx-transient/dldenoiser/enabled=false',
        '--/renderer/multiGpu/enabled':         '--/renderer/multiGpu/enabled=false',
        '--/rtx/post/aa/op':                    '--/rtx/post/aa/op=0',
        '--/rtx/reflections/enabled':           '--/rtx/reflections/enabled=false',
        '--/rtx/indirectDiffuse/enabled':       '--/rtx/indirectDiffuse/enabled=false',
        '--/rtx/translucency/enabled':          '--/rtx/translucency/enabled=false',
    }

    default_kit_args = []
    for kit_path, kit_arg in defaults_map.items():
        if kit_path not in user_kit_paths:
            default_kit_args.append(kit_arg)

    # Inject all Kit args into sys.argv
    all_kit_args = user_kit_args + default_kit_args
    for arg in all_kit_args:
        if arg not in sys.argv:
            sys.argv.append(arg)
            print(f"[EVAL][INFO] Injected Kit startup arg: {arg}")

    # ------------------------------------------------------------------
    # Step 4: Launch Isaac Lab app
    # ------------------------------------------------------------------
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # ------------------------------------------------------------------
    # Step 5: Post-startup carb settings (fallback reinforcement)
    # ------------------------------------------------------------------
    import carb
    try:
        s = carb.settings.get_settings()
        s.set("/ngx/enabled", False)
        s.set("/rtx/post/dlss/execMode", 0)
        s.set("/rtx/post/aa/op", 0)
        s.set("/rtx/reflections/enabled", False)
        s.set("/rtx/indirectDiffuse/enabled", False)
        s.set("/rtx/translucency/enabled", False)
        print("[EVAL][INFO] Applied post-startup RTX/NGX carb defaults")
    except Exception as e:
        print(f"[EVAL][WARN] Failed to apply post-startup carb defaults: {e}")

    # Apply any remaining unknown carb settings (user overrides)
    if unknown:
        apply_carb_settings(unknown)

    env = None
    try:
        # ==========================================================================
        # IMPORTS AFTER AppLauncher (required for Isaac Lab compatibility)
        # ==========================================================================
        import torch

        # RSL-RL imports with clear error message
        try:
            from rsl_rl.runners import OnPolicyRunner
        except ImportError as e:
            raise ImportError(
                "Failed to import rsl_rl. Please install the training dependencies via:\n"
                "  pip install -e '.[train]'\n"
                f"Original error: {e}"
            ) from e

        # Isaac Lab imports (version-resilient)
        try:
            from isaaclab.envs.wrappers.rsl_rl import RslRlVecEnvWrapper
        except ModuleNotFoundError:
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        # Project imports
        from pallet_rl.envs.pallet_task import PalletTask, PalletTaskCfg
        from pallet_rl.models.rsl_rl_wrapper import PalletizerActorCritic

        # ==========================================================================
        # Evaluation
        # ==========================================================================

        device = torch.device(args.device)

        # Environment configuration
        print("[EVAL] Creating environment...", flush=True)
        env_cfg = PalletTaskCfg()
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = args.device

        # Debug overrides for evaluation
        # 1. Align render_interval with decimation to avoid multiple render calls per step
        env_cfg.sim.render_interval = env_cfg.decimation
        # 2. Shorten episode length for quicker visual debugging
        if hasattr(env_cfg, "episode_length_s"):
            env_cfg.episode_length_s = min(env_cfg.episode_length_s, 5.0)

        render_mode = None if args.headless else "rgb_array"
        env = PalletTask(cfg=env_cfg, render_mode=render_mode)
        print("[EVAL] Environment created", flush=True)

        print("[EVAL] Wrapping environment for RSL-RL...", flush=True)
        env = RslRlVecEnvWrapper(env)
        print("[EVAL] Environment wrapped", flush=True)

        # Build minimal RSL-RL config for evaluation
        eval_cfg = {
            "seed": 42,
            "num_steps_per_env": 24,
            "runner": {
                "policy_class_name": "ActorCritic",
                "algorithm_class_name": "PPO",
                "num_steps_per_env": 24,
                "max_iterations": 1,
                "save_interval": 0,
                "experiment_name": "palletizer_eval",
                "run_name": "eval",
                "resume": True,
                "load_run": -1,
                "checkpoint": args.checkpoint,
                "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
            },
            "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
            "policy": {
                "class_name": "ActorCritic",
                "init_noise_std": 0.0,
                "actor_hidden_dims": [256, 128],
                "critic_hidden_dims": [256, 128],
                "activation": "elu",
            },
            "algorithm": {
                "class_name": "PPO",
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

        # Register custom policy class with RSL-RL
        print("[EVAL] Registering custom policy...", flush=True)
        from pallet_rl.models.rsl_rl_wrapper import register_custom_policy
        register_custom_policy()
        print("[EVAL] Custom policy registered.", flush=True)

        print("[EVAL] Building OnPolicyRunner...", flush=True)
        runner = OnPolicyRunner(env=env, train_cfg=eval_cfg, log_dir=args.log_dir, device=str(device))
        print("[EVAL] OnPolicyRunner built.", flush=True)

        # Load checkpoint into runner/policy
        print(f"[EVAL] Loading checkpoint: {args.checkpoint}", flush=True)
        runner.load(args.checkpoint)
        print("[EVAL] Checkpoint loaded successfully", flush=True)

        # Obtain the inference policy
        print("[EVAL] Preparing inference policy...", flush=True)
        policy = runner.get_inference_policy(device=str(device))
        print("[EVAL] Inference policy ready.", flush=True)

        # Simple evaluation loop using deterministic actions
        obs = unwrap_obs(runner.env.reset())
        episode_counts = torch.zeros(args.num_envs, dtype=torch.long, device=device)

        print(f"[EVAL] Starting evaluation rollouts (max_episodes={args.max_episodes}, num_envs={args.num_envs})...",
              flush=True)

        # rollout loop with diagnosis
        global_step = 0
        max_debug_steps = 50

        while int(episode_counts.min().item()) < args.max_episodes:
            t_start = time.time()
            with torch.no_grad():
                # Use the policy callable instead of direct actor_critic access
                actions = policy(obs)

            obs, rewards, dones, infos = runner.env.step(actions)
            obs = unwrap_obs(obs)
            t_end = time.time()
            dt = t_end - t_start

            # Diagnosis logging
            global_step += 1
            reward_mean = rewards.mean().item()
            done_any = dones.any().item()
            act_mean = actions.mean().item()
            act_std = actions.std().item()
            act_max = actions.abs().max().item()
            eps_min = int(episode_counts.min().item())

            print(f"[EVAL][STEP] step={global_step} dt_wall={dt:.3f}s reward_mean={reward_mean:.4f} "
                  f"done={done_any} episodes={eps_min} act_mean={act_mean:.4f} act_std={act_std:.4f} "
                  f"act_max={act_max:.4f}", flush=True)

            # Count completed episodes
            if "time_outs" in infos:
                done_flags = dones | infos["time_outs"]
            else:
                done_flags = dones
            episode_counts += done_flags.to(device=device, dtype=torch.long)

            if global_step >= max_debug_steps:
                print(f"[EVAL][INFO] Reached max_debug_steps={max_debug_steps} before finishing requested episodes.",
                      flush=True)
                break

        print("[EVAL] Evaluation complete.", flush=True)

    except Exception as e:
        print(f"[EVAL][FAIL] {repr(e)}", flush=True)
        traceback.print_exc()
        raise

    finally:
        print("[EVAL] Shutting down...", flush=True)
        if env is not None:
            try:
                env.close()
                print("[EVAL] Environment closed.", flush=True)
            except Exception as e_env:
                print(f"[EVAL][WARN] env.close() failed: {e_env}", flush=True)
        try:
            simulation_app.close()
            print("[EVAL] Simulation app closed.", flush=True)
        except Exception as e_app:
            print(f"[EVAL][WARN] simulation_app.close() failed: {e_app}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[EVAL][FATAL] {repr(e)}", flush=True)
        traceback.print_exc()
        raise
