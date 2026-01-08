
import os
import argparse
import sys
import torch

from omni.isaac.lab.app import AppLauncher

# 1. Launch App first
parser = argparse.ArgumentParser(description="Train Palletizer with RSL-RL")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config (Env)") # Kept for env config
parser.add_argument("--rsl_config", type=str, default="configs/rsl_rl_config.yaml", help="Path to RSL-RL config")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 2. Imports after App Launch
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs import DirectRLEnvCfg

# Import Task
from pallet_rl.envs.pallet_task import PalletTask, PalletTaskCfg
from pallet_rl.models.rsl_rl_wrapper import PalletizerActorCritic

# RSL-RL Imports
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic

# Utils
import yaml
from pallet_rl.algo.utils import load_config

def main():
    # Load RSL-RL Config
    with open(args.rsl_config, 'r') as f:
        rsl_cfg = yaml.safe_load(f)
    
    # Env Config
    # Assuming we use the class default or load overrides
    env_cfg = PalletTaskCfg()
    # Apply overrides from args.config if needed, but for now using defaults/class
    env_cfg.num_envs = 4096
    
    # Create Environment
    env = PalletTask(cfg=env_cfg, render_mode="rgb_array" if args.headless else None)
    
    # Wrap for RSL-RL
    # RSL-RL OnPolicyRunner expects a VecEnv-like object with:
    # get_observations(), step(), reset(), property num_envs, num_obs, num_actions
    # DirectRLEnv provides these but interface matches Gym (get_obs returns dict).
    # We need a wrapper to flatten dict to tensor corresponding to "policy".
    
    class RslRlWrapper:
        def __init__(self, env):
            self.env = env
            self.num_envs = env.num_envs
            # RSL-RL expects num_obs and num_critic_obs (can be same)
            # We defined these in config, but Env knows truth?
            # Env returns 4120.
            self.num_obs = env.cfg.num_observations
            self.num_critic_obs = self.num_obs
            self.num_actions = env.cfg.num_actions
            
            # Device
            self.device = env.device

        def get_observations(self):
            obs_dict = self.env._get_observations() # calling internal as it returns dict directly? 
            # Or env.reset()? 
            # DirectRLEnv step returns (obs, rew, ...). Obs is dict.
            # We should probably use public API.
            # But DirectRLEnv public API returns dict. RSL-RL wants tensor.
            # Let's check if we can get current obs.
            # wrapper not standard? 
            # RSL-RL runner calls env.get_observations().
            return obs_dict["policy"], obs_dict.get("critic", obs_dict["policy"])

        def step(self, actions):
            obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
            dones = terminated | truncated
            return obs_dict["policy"], obs_dict.get("critic", obs_dict["policy"]), rew, dones, extras

        def reset(self):
            obs_dict, _ = self.env.reset()
            return obs_dict["policy"], obs_dict.get("critic", obs_dict["policy"])
            
        @property
        def episode_length_buf(self):
            return self.env.episode_length_buf # Access underlying buffer if available

    wrapped_env = RslRlWrapper(env)
    
    # Create Runner
    # Ensure log dir
    log_dir = "runs/rsl_rl_palletizer"
    
    # Inject our Custom Policy Class into RSL-RL module registry OR pass it directly?
    # RSL-RL instantiates by class name string usually.
    # We need to register it or ensure it's importable.
    # Or we can patch it?
    # RSL-RL runner.py: `class_ = getattr(modules, policy_class_name)`
    # It looks in rsl_rl.modules.
    # We need to make PalletizerActorCritic available there OR modify runner to search elsewhere.
    # The prompt says: "Ensure the policy section points to your PalletizerActorCritic."
    # AND "Import rsl_rl.modules.ActorCritic".
    # I will dynamically inject it into rsl_rl.modules so the string lookup works.
    import rsl_rl.modules
    rsl_rl.modules.PalletizerActorCritic = PalletizerActorCritic
    
    runner = OnPolicyRunner(wrapped_env, rsl_cfg, log_dir=log_dir, device="cuda:0")
    
    # Learn
    runner.learn(num_learning_iterations=rsl_cfg["runner"]["max_iterations"], init_at_random_ep_len=True)
    
    env.close()

if __name__ == "__main__":
    main()
