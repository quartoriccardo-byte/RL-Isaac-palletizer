
import torch

class RolloutBuffer:
    def __init__(self, num_steps, num_envs, obs_shape, device):
        self.obs = torch.zeros((num_steps, num_envs, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((num_steps, num_envs, 4), dtype=torch.long, device=device)  # pick,yaw,x,y
        self.rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)
        self.values = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.logprobs = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.step = 0

    def add(self, obs, action, reward, done, value, logprob):
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value
        self.logprobs[self.step] = logprob
        self.step += 1

    def reset(self):
        self.step = 0
