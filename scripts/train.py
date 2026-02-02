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
import re
import sys

# =============================================================================
# Step 1: Parse arguments (no side effects at import time)
# =============================================================================

def parse_args():
    """Parse command line arguments, allowing unknown Kit/Carb settings.
    
    Returns:
        tuple: (args, unknown) where args is the Namespace of known args,
               and unknown is a list of unrecognized arguments (e.g., --/rtx/...).
    """
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
    
    # Use parse_known_args to accept unknown Kit/Carb settings (--/path=value)
    args, unknown = parser.parse_known_args()
    return args, unknown


def parse_carb_setting(value_str: str):
    """Parse a carb setting value string to appropriate Python type.
    
    Converts:
        - 'true'/'false' -> bool
        - integer strings -> int
        - float strings -> float
        - otherwise -> str
    """
    # Boolean
    if value_str.lower() == 'true':
        return True
    if value_str.lower() == 'false':
        return False
    
    # Integer
    try:
        return int(value_str)
    except ValueError:
        pass
    
    # Float
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # String fallback
    return value_str


def apply_carb_settings(unknown_args: list):
    """Apply Kit/Carb settings from unknown arguments.
    
    Parses arguments of the form --/some/path=value and applies them
    via carb.settings.get_settings().set().
    
    Args:
        unknown_args: List of unrecognized CLI arguments.
    """
    import carb
    settings = carb.settings.get_settings()
    
    # Pattern: --/some/path=value
    pattern = re.compile(r'^--(/[^=]+)=(.*)$')
    
    for arg in unknown_args:
        match = pattern.match(arg)
        if match:
            path = match.group(1)
            value_str = match.group(2)
            value = parse_carb_setting(value_str)
            settings.set(path, value)
            print(f"[INFO] Applied carb setting: {path} = {value}")


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    """Main training entry point."""
    # Parse arguments (inside main to avoid import-time side effects)
    # unknown contains Kit/Carb settings like --/rtx/post/dlss/execMode=0
    args, unknown = parse_args()
    
    # Launch Isaac Lab app (MUST be before other imports that touch simulation)
    from isaaclab.app import AppLauncher
    
    # Try to pass unknown args to AppLauncher if it supports extra_args parameter
    # This allows Kit settings to be applied during app initialization
    import inspect
    launcher_sig = inspect.signature(AppLauncher.__init__)
    if 'extra_args' in launcher_sig.parameters:
        app_launcher = AppLauncher(args, extra_args=unknown)
    else:
        # Fallback: AppLauncher doesn't support extra_args, just pass known args
        app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    
    # Apply carb settings from unknown args (guaranteed fallback that always works)
    # This ensures Kit settings are applied even if AppLauncher didn't handle them
    if unknown:
        apply_carb_settings(unknown)
    
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
        # Different rsl_rl versions expect obs_groups at different locations:
        # - Some expect it nested under runner_cfg["obs_groups"]
        # - Some expect it at top-level cfg["obs_groups"]
        # We ensure it exists at BOTH locations for maximum compatibility.
        # =======================================================================
        default_obs_groups = {
            "policy": ["policy"],
            "critic": ["critic"],
        }
        # Check both locations for existing obs_groups
        obs_groups = runner_cfg.get("obs_groups") or cfg.get("obs_groups") or default_obs_groups
        # Set at BOTH locations
        runner_cfg["obs_groups"] = obs_groups
        cfg["obs_groups"] = obs_groups
        
        # =======================================================================
        # REQUIRED: class_name inside policy and algorithm sections
        # rsl_rl OnPolicyRunner expects:
        #   - cfg["policy"]["class_name"] (e.g., "ActorCritic")
        #   - cfg["algorithm"]["class_name"] (e.g., "PPO")
        # Our YAML stores these under runner as policy_class_name / algorithm_class_name.
        # Inject them into policy/algorithm sections for compatibility.
        # =======================================================================
        policy_cfg = cfg.setdefault("policy", {})
        algo_cfg = cfg.setdefault("algorithm", {})
        
        # Get class names from runner section (fallback to defaults)
        policy_class = runner_cfg.get("policy_class_name", "ActorCritic")
        algo_class = runner_cfg.get("algorithm_class_name", "PPO")
        
        # Inject class_name into policy and algorithm sections
        policy_cfg.setdefault("class_name", policy_class)
        algo_cfg.setdefault("class_name", algo_class)
        
        # =======================================================================
        # Mirror key runner fields to top-level for rsl_rl versions that expect flat config
        # =======================================================================
        mirror_keys = [
            "max_iterations", "num_steps_per_env", "save_interval",
            "experiment_name", "run_name", "resume", "checkpoint", "load_run",
            "policy_class_name", "algorithm_class_name", "class_name",
        ]
        for k in mirror_keys:
            if k in runner_cfg:
                cfg.setdefault(k, runner_cfg[k])

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
    # Step 2: Determine render mode (video recording requires rgb_array)
    # -------------------------------------------------------------------------
    # If video recording is requested, force enable cameras and use rgb_array
    if args.video:
        args.enable_cameras = True
        render_mode = "rgb_array"
        print(f"[INFO] Video recording enabled: render_mode='rgb_array', cameras forced ON")
    else:
        render_mode = None if args.headless else "rgb_array"
    
    # -------------------------------------------------------------------------
    # Step 3: Create environment
    # -------------------------------------------------------------------------
    print("Creating environment...")
    env = PalletTask(cfg=env_cfg, render_mode=render_mode)
    
    # -------------------------------------------------------------------------
    # Step 4: Apply RecordVideo wrapper (if video recording is requested)
    # -------------------------------------------------------------------------
    # IMPORTANT: RecordVideo must wrap the base Gymnasium env BEFORE RslRlVecEnvWrapper
    if args.video:
        from gymnasium.wrappers import RecordVideo
        
        video_folder = os.path.join(args.log_dir, args.experiment_name, "videos")
        os.makedirs(video_folder, exist_ok=True)
        
        # step_trigger: record every video_interval steps for video_length frames
        step_trigger = lambda step: step % args.video_interval == 0
        
        env = RecordVideo(
            env,
            video_folder=video_folder,
            step_trigger=step_trigger,
            video_length=args.video_length,
            name_prefix="rl-video",
        )
        print(f"[INFO] Recording videos to: {video_folder}")
    
    # -------------------------------------------------------------------------
    # Step 5: Wrap for RSL-RL
    # -------------------------------------------------------------------------
    print("Wrapping environment for RSL-RL...")
    env = RslRlVecEnvWrapper(env)
    
    # -------------------------------------------------------------------------
    # Step 6: Inject custom policy class
    # -------------------------------------------------------------------------
    # RSL-RL uses module-level lookup for policy class
    # We monkey-patch to use our custom CNN-based policy
    import rsl_rl.modules
    rsl_rl.modules.ActorCritic = PalletizerActorCritic
    
    # -------------------------------------------------------------------------
    # Step 7: Create RSL-RL runner
    # -------------------------------------------------------------------------
    print("Initializing RSL-RL runner...")
    rsl_cfg = get_rsl_rl_cfg(args)
    
    # DEBUG: Verify obs_groups is present at required locations
    print(f"[DEBUG] rsl_cfg top-level keys: {list(rsl_cfg.keys())}")
    print(f"[DEBUG] top obs_groups: {rsl_cfg.get('obs_groups')}")
    print(f"[DEBUG] runner obs_groups: {rsl_cfg.get('runner', {}).get('obs_groups')}")
    print(f"[DEBUG] policy.class_name: {rsl_cfg.get('policy', {}).get('class_name')}")
    print(f"[DEBUG] algorithm.class_name: {rsl_cfg.get('algorithm', {}).get('class_name')}")
    
    runner = OnPolicyRunner(
        env=env,
        train_cfg=rsl_cfg,
        log_dir=args.log_dir,
        device=args.device
    )
    
    # -------------------------------------------------------------------------
    # Step 8: Load checkpoint if resuming
    # -------------------------------------------------------------------------
    if args.resume and args.checkpoint is not None:
        print(f"Resuming from checkpoint: {args.checkpoint}")
        runner.load(args.checkpoint)
    
    # -------------------------------------------------------------------------
    # Step 9: Train!
    # -------------------------------------------------------------------------
    print(f"\nStarting training for {args.max_iterations} iterations...\n")
    
    runner.learn(
        num_learning_iterations=args.max_iterations,
        init_at_random_ep_len=True
    )
    
    # -------------------------------------------------------------------------
    # Step 10: Cleanup
    # -------------------------------------------------------------------------
    print("\nTraining complete. Shutting down...")
    
    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()

