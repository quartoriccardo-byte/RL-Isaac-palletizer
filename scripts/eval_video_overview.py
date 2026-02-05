"""
Isaac Lab 5.0+ Evaluation Video Overview Script

Records a tiled mosaic video showing multiple parallel environments
during policy evaluation. Each environment's camera view is composed
into a single grid video for visual inspection of learned behavior.

Usage:
    python scripts/eval_video_overview.py \
        --headless \
        --checkpoint path/to/model.pt \
        --num_envs 8 \
        --tile_rows 2 \
        --tile_cols 4 \
        --video_folder runs/mosaic_eval
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import gymnasium as gym
import numpy as np


# =============================================================================
# Step 1: Parse arguments (no side effects at import time)
# =============================================================================

def parse_args():
    """Parse command line arguments, allowing unknown Kit/Carb settings.
    
    Returns:
        tuple: (args, unknown) where args is the Namespace of known args,
               and unknown is a list of unrecognized arguments (e.g., --/rtx/...).
    """
    parser = argparse.ArgumentParser(description="Record Evaluation Mosaic Video")
    
    # Simulation
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of environments")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device")
    
    # Checkpoint
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to RSL-RL checkpoint (.pt)")
    
    # Evaluation
    parser.add_argument("--eval_steps", type=int, default=500, help="Rollout steps for video")
    
    # Video recording
    parser.add_argument("--video", action="store_true", default=True, help="Enable video recording (default: True)")
    parser.add_argument("--video_length", type=int, default=500, help="Video length in steps")
    parser.add_argument("--video_folder", type=str, default="runs/eval_videos", help="Video output folder")
    
    # Mosaic configuration
    parser.add_argument("--tile_rows", type=int, default=2, help="Number of tile rows")
    parser.add_argument("--tile_cols", type=int, default=4, help="Number of tile columns")
    parser.add_argument("--cam_width", type=int, default=320, help="Per-env camera width")
    parser.add_argument("--cam_height", type=int, default=180, help="Per-env camera height")
    
    # Isaac Lab launcher args
    parser.add_argument("--livestream", type=int, default=0, help="Livestream mode")
    parser.add_argument("--enable_cameras", action="store_true", help="Enable camera sensors")
    
    args, unknown = parser.parse_known_args()
    return args, unknown


def parse_carb_setting(value_str: str):
    """Parse a carb setting value string to appropriate Python type."""
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


def apply_carb_settings(unknown_args: list):
    """Apply Kit/Carb settings from unknown arguments."""
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
            print(f"[INFO] Applied carb setting: {path} = {value}")


# =============================================================================
# Main Evaluation Loop
# =============================================================================

def main():
    """Main evaluation entry point."""
    args, unknown = parse_args()
    
    # Force enable cameras for mosaic recording
    args.enable_cameras = True
    args.video = True
    
    # =========================================================================
    # CRITICAL: Inject Kit/Carb args into sys.argv BEFORE importing AppLauncher
    # =========================================================================
    user_kit_args = [arg for arg in unknown if arg.startswith('--/')]
    user_kit_paths = {arg.split('=')[0] for arg in user_kit_args}
    
    # Default Kit args for headless video mode
    default_kit_args = []
    defaults_map = {
        '--/ngx/enabled': '--/ngx/enabled=false',
        '--/rtx/post/dlss/enabled': '--/rtx/post/dlss/enabled=false',
        '--/rtx/post/dlss/execMode': '--/rtx/post/dlss/execMode=0',
        '--/rtx/post/aa/op': '--/rtx/post/aa/op=0',
        '--/rtx-transient/dlssg/enabled': '--/rtx-transient/dlssg/enabled=false',
        '--/rtx-transient/dldenoiser/enabled': '--/rtx-transient/dldenoiser/enabled=false',
        '--/renderer/multiGpu/enabled': '--/renderer/multiGpu/enabled=false',
        '--/rtx/translucency/enabled': '--/rtx/translucency/enabled=false',
        '--/rtx/reflections/enabled': '--/rtx/reflections/enabled=false',
        '--/rtx/indirectDiffuse/enabled': '--/rtx/indirectDiffuse/enabled=false',
    }
    for kit_path, kit_arg in defaults_map.items():
        if kit_path not in user_kit_paths:
            default_kit_args.append(kit_arg)
    
    all_kit_args = user_kit_args + default_kit_args
    for arg in all_kit_args:
        if arg not in sys.argv:
            sys.argv.append(arg)
            print(f"[INFO] Injected Kit startup arg: {arg}")
    
    # Now import and launch Isaac Lab app
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    
    # Post-startup carb settings (fallback reinforcement)
    import carb
    try:
        s = carb.settings.get_settings()
        s.set("/ngx/enabled", False)
        s.set("/rtx/post/dlss/execMode", 0)
        s.set("/rtx/ambientOcclusion/enabled", False)
        s.set("/rtx/reflections/enabled", False)
        s.set("/rtx/translucency/enabled", False)
        s.set("/rtx/indirectDiffuse/enabled", False)
        s.set("/rtx/post/aa/op", 0)
        print("[INFO] Applied post-startup RTX/DLSS carb settings")
    except Exception as e:
        print(f"[WARN] Failed to apply RTX/DLSS settings: {e}")
    
    if unknown:
        apply_carb_settings(unknown)
    
    # ==========================================================================
    # IMPORTS AFTER AppLauncher (required for Isaac Lab compatibility)
    # ==========================================================================
    import torch
    import yaml

    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError as e:
        raise ImportError(
            "Failed to import rsl_rl. Please install RSL-RL:\n"
            "  pip install git+https://github.com/leggedrobotics/rsl_rl.git\n"
            f"Original error: {e}"
        ) from e

    try:
        from isaaclab.envs.wrappers.rsl_rl import RslRlVecEnvWrapper
    except ModuleNotFoundError:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from pallet_rl.envs.pallet_task import PalletTask, PalletTaskCfg
    from pallet_rl.models.rsl_rl_wrapper import PalletizerActorCritic

    # =========================================================================
    # MosaicOverviewWrapper: Compose multi-env frames into a tiled grid
    # =========================================================================
    class MosaicOverviewWrapper(gym.Wrapper):
        """Wrapper that composes all env camera frames into a single mosaic.
        
        Reads RGB buffers from all environments' cameras and tiles them
        into a (rows*H, cols*W, 3) grid for video recording.
        """
        
        def __init__(self, env, tile_rows: int, tile_cols: int):
            super().__init__(env)
            self.tile_rows = tile_rows
            self.tile_cols = tile_cols
            self._frame_count = 0
        
        def render(self):
            """Compose all env camera frames into a mosaic."""
            # Force sim render tick
            if hasattr(self.env, 'sim') and self.env.sim is not None:
                self.env.sim.render()
            
            # Get camera sensor
            if "camera" not in self.env.scene.keys():
                print(f"[MOSAIC ERR] No camera in scene!")
                return None
            
            try:
                camera = self.env.scene["camera"]
                camera.update(dt=self.env.step_dt)
                
                # Get RGB: shape (num_envs, H, W, 3/4)
                rgb_data = camera.data.output.get("rgb")
                if rgb_data is None:
                    print("[MOSAIC DBG] rgb_data is None")
                    return None
                
                # Convert to numpy
                if hasattr(rgb_data, 'cpu'):
                    rgb_np = rgb_data.cpu().numpy()
                else:
                    rgb_np = np.array(rgb_data)
                
                # Drop alpha if present
                if rgb_np.shape[-1] == 4:
                    rgb_np = rgb_np[..., :3]
                
                # Convert float [0,1] to uint8 if needed
                if rgb_np.dtype in (np.float32, np.float64):
                    rgb_np = (np.clip(rgb_np, 0.0, 1.0) * 255).astype(np.uint8)
                
                # Compose mosaic: (num_envs, H, W, 3) -> (rows*H, cols*W, 3)
                num_envs, h, w, c = rgb_np.shape
                rows = self.tile_rows
                cols = self.tile_cols
                
                # Pad with black if num_envs < rows*cols
                total_tiles = rows * cols
                if num_envs < total_tiles:
                    padding = np.zeros((total_tiles - num_envs, h, w, c), dtype=rgb_np.dtype)
                    rgb_np = np.concatenate([rgb_np, padding], axis=0)
                elif num_envs > total_tiles:
                    rgb_np = rgb_np[:total_tiles]
                
                # Reshape: (rows, cols, H, W, C) -> (rows*H, cols*W, C)
                rgb_np = rgb_np.reshape(rows, cols, h, w, c)
                rgb_np = rgb_np.transpose(0, 2, 1, 3, 4)  # (rows, H, cols, W, C)
                mosaic = rgb_np.reshape(rows * h, cols * w, c)
                
                self._frame_count += 1
                if self._frame_count % 50 == 1:
                    print(f"[MOSAIC] Frame {self._frame_count}: shape={mosaic.shape} "
                          f"min={mosaic.min()} max={mosaic.max()}")
                
                return mosaic
                
            except Exception as e:
                print(f"[MOSAIC ERR] render() failed: {e}")
                import traceback
                traceback.print_exc()
                return None

    # =========================================================================
    # RenderNormWrapper: Ensure uint8 RGB output for RecordVideo
    # =========================================================================
    class RenderNormWrapper(gym.Wrapper):
        """Ensure render() returns uint8 RGB numpy array."""
        
        def render(self):
            frame = self.env.render()
            if frame is None:
                return None
            
            if hasattr(frame, 'cpu'):
                frame = frame.cpu().numpy()
            
            if frame.dtype in (np.float32, np.float64):
                frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
            
            return frame

    # ==========================================================================
    # Evaluation Setup
    # ==========================================================================

    print(f"\n{'='*60}")
    print("Isaac Lab 5.0+ Mosaic Evaluation Video")
    print(f"{'='*60}")
    print(f"Device: {args.device}")
    print(f"Environments: {args.num_envs}")
    print(f"Mosaic: {args.tile_rows} x {args.tile_cols}")
    print(f"Camera: {args.cam_width} x {args.cam_height}")
    print(f"Eval steps: {args.eval_steps}")
    print(f"Video folder: {args.video_folder}")
    print(f"{'='*60}\n")

    env = None
    try:
        # Create environment configuration
        env_cfg = PalletTaskCfg()
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = args.device
        env_cfg.sim.render_interval = 1  # Force render every step
        
        # Override camera resolution for tiling
        env_cfg.scene.camera.width = args.cam_width
        env_cfg.scene.camera.height = args.cam_height
        
        render_mode = "rgb_array"
        
        print("Creating environment...")
        env = PalletTask(cfg=env_cfg, render_mode=render_mode)
        
        # Camera warm-up
        print("[INFO] Warming up camera with 10 render passes...")
        obs, _ = env.reset()
        for i in range(10):
            action = torch.zeros(env.num_envs, 5, device=args.device)
            obs, _, _, _, _ = env.step(action)
            frame = env.render()
            if frame is not None and i % 3 == 0:
                print(f"  Warm-up {i+1}/10: shape={frame.shape}")
        print("[INFO] Camera warm-up complete")
        
        # Apply wrappers
        # 1. MosaicOverviewWrapper - compose multi-env frames
        print(f"Applying MosaicOverviewWrapper ({args.tile_rows}x{args.tile_cols})...")
        env = MosaicOverviewWrapper(env, tile_rows=args.tile_rows, tile_cols=args.tile_cols)
        
        # 2. RenderNormWrapper - ensure uint8 RGB
        env = RenderNormWrapper(env)
        
        # 3. RecordVideo - single clip (trigger at step 0)
        from gymnasium.wrappers import RecordVideo
        
        os.makedirs(args.video_folder, exist_ok=True)
        step_trigger = lambda step: step == 0  # One clip at start
        
        env = RecordVideo(
            env,
            video_folder=args.video_folder,
            step_trigger=step_trigger,
            video_length=args.video_length,
            name_prefix="rl-video",
        )
        print(f"[INFO] Recording video to: {args.video_folder}")
        
        # 4. RslRlVecEnvWrapper for RSL-RL
        print("Wrapping environment for RSL-RL...")
        env = RslRlVecEnvWrapper(env)
        
        # Inject custom policy class
        import rsl_rl.modules
        rsl_rl.modules.ActorCritic = PalletizerActorCritic
        
        # Build minimal RSL-RL config for evaluation
        cfg_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "pallet_rl",
            "configs",
            "rsl_rl_config.yaml",
        )
        cfg_path = os.path.abspath(cfg_path)
        
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                rsl_cfg = yaml.safe_load(f)
        else:
            # Fallback minimal config
            rsl_cfg = {
                "seed": 42,
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
        
        # Set runner config for evaluation
        runner_cfg = rsl_cfg.setdefault("runner", {})
        runner_cfg["resume"] = True
        runner_cfg["checkpoint"] = args.checkpoint
        runner_cfg["max_iterations"] = 1
        runner_cfg["save_interval"] = 0
        
        # Ensure required fields
        obs_groups = {"policy": ["policy"], "critic": ["critic"]}
        runner_cfg["obs_groups"] = obs_groups
        rsl_cfg["obs_groups"] = obs_groups
        
        policy_cfg = rsl_cfg.setdefault("policy", {})
        policy_cfg.setdefault("class_name", "ActorCritic")
        
        algo_cfg = rsl_cfg.setdefault("algorithm", {})
        algo_cfg.setdefault("class_name", "PPO")
        
        print("Initializing RSL-RL runner...")
        runner = OnPolicyRunner(
            env=env,
            train_cfg=rsl_cfg,
            log_dir=args.video_folder,
            device=args.device
        )
        
        # Load checkpoint
        print(f"Loading checkpoint: {args.checkpoint}")
        runner.load(args.checkpoint)
        
        # Evaluation loop
        print(f"\nStarting evaluation for {args.eval_steps} steps...\n")
        
        obs = runner.env.reset()
        
        for step in range(args.eval_steps):
            with torch.no_grad():
                actions = runner.alg.actor_critic.act_inference(obs["policy"])
            
            obs, rewards, dones, infos = runner.env.step(actions)
            
            if step % 100 == 0:
                print(f"Step {step}/{args.eval_steps}")
        
        print("\nEvaluation complete.")
        print(f"Video saved to: {args.video_folder}")
        
    finally:
        print("\nShutting down...")
        
        if env is not None:
            try:
                env.close()
                print("[INFO] Environment closed successfully.")
            except Exception as e:
                print(f"[WARN] env.close() failed: {e}")
        
        try:
            simulation_app.close()
            print("[INFO] Simulation app closed successfully.")
        except Exception as e:
            print(f"[WARN] simulation_app.close() failed: {e}")


if __name__ == "__main__":
    main()
