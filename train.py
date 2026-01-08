
import os
import argparse
import sys
import torch

from omni.isaac.lab.app import AppLauncher

# 1. Launch App first
parser = argparse.ArgumentParser(description="Train Palletizer with RSL-RL")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config (Env)") 
parser.add_argument("--rsl_config", type=str, default="configs/rsl_rl_config.yaml", help="Path to RSL-RL config")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 2. Imports after App Launch
from omni.isaac.lab.utils import configclass
# Using standard wrapper location for Isaac Lab
# Try importing from typical locations
try:
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
except ImportError:
    try:
        from omni.isaac.lab.wrappers.rsl_rl import RslRlVecEnvWrapper
    except ImportError:
        # Fallback to local definition if official one moves/is missing
        from omni.isaac.lab.envs import DirectRLEnv
        class RslRlVecEnvWrapper:
             def __init__(self, env):
                self.env = env
                self.num_envs = env.num_envs
                self.num_obs = env.cfg.num_observations
                self.num_critic_obs = self.num_obs
                self.num_actions = env.cfg.num_actions
                self.device = env.device

             def get_observations(self):
                obs = self.env._get_observations()
                return obs["policy"], obs.get("critic", obs["policy"])

             def step(self, actions):
                obs, rew, terminated, truncated, extras = self.env.step(actions)
                dones = terminated | truncated
                return obs["policy"], obs.get("critic", obs["policy"]), rew, dones, extras

             def reset(self):
                obs, _ = self.env.reset()
                return obs["policy"], obs.get("critic", obs["policy"])
             
             @property
             def episode_length_buf(self):
                return self.env.episode_length_buf

# Import Task
from pallet_rl.envs.pallet_task import PalletTask, PalletTaskCfg
from pallet_rl.models.rsl_rl_wrapper import PalletizerActorCritic

# RSL-RL Imports
from rsl_rl.runners import OnPolicyRunner

# Utils
import yaml

def main():
    # Load RSL-RL Config
    with open(args.rsl_config, 'r') as f:
        rsl_cfg = yaml.safe_load(f)
    
    # Env Config
    env_cfg = PalletTaskCfg()
    env_cfg.num_envs = 4096
    
    # Create Environment
    env = PalletTask(cfg=env_cfg, render_mode="rgb_array" if args.headless else None)
    
    # Wrap
    env = RslRlVecEnvWrapper(env)
    
    # Inject Custom Policy
    import rsl_rl.modules
    rsl_rl.modules.PalletizerActorCritic = PalletizerActorCritic
    
    # Runner
    log_dir = "runs/rsl_rl_palletizer"
    runner = OnPolicyRunner(env, rsl_cfg, log_dir=log_dir, device=env.device)
    
    # Learn
    runner.learn(num_learning_iterations=rsl_cfg["runner"]["max_iterations"], init_at_random_ep_len=True)
    
    # Close
    # env.close() # RslRlVecEnvWrapper might not have close, access inner?
    # env.env.close() if manual wrapper
    
    if simulation_app:
        simulation_app.close()

if __name__ == "__main__":
    main()
