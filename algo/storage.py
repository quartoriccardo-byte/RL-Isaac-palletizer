
import torch

class RolloutBuffer:
    def __init__(self, num_steps, num_envs, obs_shape, device):
        self.obs = torch.zeros((num_steps, num_envs, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((num_steps, num_envs, 4), dtype=torch.long, device=device)  # pick,yaw,x,y
        self.rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)
        self.values = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.logprobs = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        # Assuming mask is (L*W) boolean/int? Or full mask?
        # User constraint: "Action Masking... apply -1e8".
        # We need to store standard mask to pass to forward().
        # Mask shape is (N, L*W). L*W depends on obs_shape? No, hardcoded grid.
        # But here we don't know L,W directly?
        # We will infer or pass it?
        # Better: let's store it as tensor of shape (num_steps, num_envs, mask_dim)
        # BUT we don't know mask_dim here unless passed.
        # Let's lazily init or pass mask_dim?
        # Or just use dynamic?
        # Let's assume mask is passed in `add` and we init `self.masks` based on first add or pass shape in init.
        # Let's blindly guess L*W = 40*48 = 1920? No strictly config dependent.
        # I'll rely on broadcasting or init in `add`? No pre-allocation is better.
        # I'll update __init__ to take mask_shape.
        pass

    def init_masks(self, mask_shape):
         self.masks = torch.zeros((self.obs.shape[0], self.obs.shape[1], *mask_shape), dtype=torch.bool, device=self.obs.device)

    def add(self, obs, action, reward, done, value, logprob, mask=None):
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value
        self.logprobs[self.step] = logprob
        if mask is not None:
             self.masks[self.step] = mask
        self.step += 1

    def reset(self):
        self.step = 0
