
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from pallet_rl.models.encoder2d import Encoder2D
from pallet_rl.models.policy_heads import SpatialPolicyHead

class ActorCritic(nn.Module):
    def __init__(self, in_channels, encoder_features, num_rotations=4):
        super().__init__()
        self.encoder = Encoder2D(in_channels=in_channels, features=encoder_features)
        # Policy Head (Actor)
        self.actor = SpatialPolicyHead(in_channels=encoder_features, num_rotations=num_rotations)
        # Value Head (Critic) - needs to process encoder features to scalar
        # Assuming encoder outputs (B, Features, H, W), we probably need to average pool or flatten for value
        self.critic_head = nn.Sequential(
             nn.AdaptiveAvgPool2d(1),
             nn.Flatten(),
             nn.Linear(encoder_features, 64),
             nn.ReLU(),
             nn.Linear(64, 1)
        )

    def forward(self, x, mask=None):
        features = self.encoder(x)
        logits = self.actor(features, mask=mask)
        value = self.critic_head(features).squeeze(-1)
        return logits, value

    def get_value(self, x):
        features = self.encoder(x)
        value = self.critic_head(features).squeeze(-1)
        return value

class PPO:
    def __init__(self, config, obs_shape):
        self.cfg = config
        self.device = torch.device(config["device"])
        
        in_channels = obs_shape[0]
        enc_features = config["model"]["encoder_features"]
        
        self.policy = ActorCritic(in_channels, enc_features).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config["algo"]["learning_rate"])
        
        self.mse_loss = nn.MSELoss()

    def act(self, obs, mask=None):
        """
        Sample action for rollout.
        obs: (B, C, H, W)
        mask: (B, L*W*rots) or similar, broadcasting support
        """
        self.policy.eval()
        with torch.no_grad():
            logits, value = self.policy(obs, mask)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action, log_prob, value

    def update(self, rollouts):
        """
        Update policy using PPO.
        rollouts: dict containing obs, actions, logprobs, returns, advantages, masks
        """
        self.policy.train()
        
        obs = rollouts["obs"]
        actions = rollouts["actions"]
        old_log_probs = rollouts["logprobs"]
        advantages = rollouts["advantages"]
        returns = rollouts["returns"]
        masks = rollouts["masks"]
        
        batch_size = self.cfg["algo"]["batch_size"]
        dataset_size = obs.size(0)
        
        clip_range = self.cfg["algo"]["clip_range"]
        ent_coef = self.cfg["algo"]["entropy_coef"]
        vf_coef = self.cfg["algo"]["value_loss_coef"]
        max_grad_norm = self.cfg["algo"]["max_grad_norm"]
        
        total_loss = 0
        
        for _ in range(self.cfg["algo"]["n_epochs"]):
            indices = torch.randperm(dataset_size, device=self.device)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]
                
                b_obs = obs[idx]
                b_actions = actions[idx]
                b_old_log_probs = old_log_probs[idx]
                b_advantages = advantages[idx]
                b_returns = returns[idx]
                b_masks = masks[idx] if masks is not None else None
                
                logits, values = self.policy(b_obs, mask=b_masks)
                dist = Categorical(logits=logits)
                
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()
                
                ratio = (new_log_probs - b_old_log_probs).exp()
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = self.mse_loss(values, b_returns) * vf_coef
                
                loss = policy_loss + value_loss - ent_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                
        return total_loss / (dataset_size / batch_size * self.cfg["algo"]["n_epochs"])
