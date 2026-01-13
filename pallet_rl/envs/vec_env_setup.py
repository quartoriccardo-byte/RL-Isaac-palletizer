from typing import Any, Dict, Tuple, List
import numpy as np
import os

class DummyVecEnv:
    """
    A pure Python dummy environment for testing the training loop on Windows/Linux
    without Isaac Lab.
    """
    def __init__(self, num_envs: int, obs_shape: Tuple, action_space: Dict):
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_space = action_space
        self.L = obs_shape[1]
        self.W = obs_shape[2]
        
        # Internal state (just random heightmaps for testing)
        self.heights = np.zeros((num_envs, self.L, self.W), dtype=np.float32)

    def reset(self):
        self.heights.fill(0.0)
        return self._get_obs()

    def step(self, actions):
        # actions: (N, 4) -> [pick, yaw, x, y]
        # Simple heuristic: increase height at action location
        if actions is not None:
            actions = np.array(actions)
            for i in range(self.num_envs):
                x = int(actions[i, 2])
                y = int(actions[i, 3])
                # Clip to bounds
                x = max(0, min(self.L-1, x))
                y = max(0, min(self.W-1, y))
                self.heights[i, x, y] += 1.0
        
        obs = self._get_obs()
        # Random reward for testing
        rewards = np.random.randn(self.num_envs).astype(np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]
        return obs, rewards, dones, infos

    def _get_obs(self):
        # Return random observation matching shape
        # Channels: 8 (proxy) + 5 (box) = 13
        obs = np.zeros((self.num_envs, *self.obs_shape), dtype=np.float32)
        # Fill height channel (channel 0) with internal state
        obs[:, 0, :, :] = self.heights
        return obs

    def get_action_mask(self):
        """
        Returns a mask of valid actions. For dummy env, all 1.0 (valid).
        Shape: (num_envs, L*W)
        """
        mask = np.ones((self.num_envs, self.L * self.W), dtype=np.float32)
        return mask

def make_vec_env(config: Dict):
    """
    Factory function to create the vectorized environment.
    """
    use_isaac = bool(int(os.environ.get("USE_ISAACLAB", "0")))
    
    if use_isaac:
        try:
            from .isaaclab_task import IsaacLabVecEnv
            return IsaacLabVecEnv(config)
        except ImportError as e:
            print(f"Warning: Failed to import IsaacLabVecEnv: {e}")
            print("Falling back to DummyVecEnv.")
    
    # Fallback or default to Dummy
    L, W = config["env"]["grid"]
    obs_shape = (8 + 5, L, W)
    action_space = {
        "pick": config["env"]["buffer_N"],
        "yaw": len(config["env"]["yaw_orients"]),
        "x": L,
        "y": W
    }
    
    print(f"Created DummyVecEnv with {config['env']['num_envs']} envs.")
    return DummyVecEnv(config["env"]["num_envs"], obs_shape, action_space)
