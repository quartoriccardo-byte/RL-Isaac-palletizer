
import os, argparse, time
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from pallet_rl.algo.utils import load_config
from pallet_rl.envs.vec_env_setup import make_vec_env
from pallet_rl.models.encoder2d import Encoder2D
from pallet_rl.models.unet2d import UNet2D
from pallet_rl.models.policy_heads import PolicyHeads
from pallet_rl.algo.storage import RolloutBuffer
from pallet_rl.algo.ppo import ppo_update

class ActorCritic(torch.nn.Module):
    def __init__(self, cfg, in_ch:int):
        super().__init__()
        L, W = cfg["env"]["grid"]
        base = cfg["model"]["encoder2d"]["base_channels"]
        self.encoder = Encoder2D(in_ch=in_ch, base=base)
        self.unet = UNet2D(in_ch=in_ch, base=cfg["model"]["unet2d"]["base_channels"])
        self.heads = PolicyHeads(self.encoder.out_ch, (L, W),
                                 n_pick=cfg["env"]["buffer_N"],
                                 n_yaw=len(cfg["env"]["yaw_orients"]),
                                 hidden=cfg["model"]["policy_heads"]["hidden"])
        self.gating_lambda = cfg["mask"]["gating_lambda"]

    def forward_policy(self, obs):
        enc = self.encoder(obs)
        mask = self.unet(obs)
        logits_pick, logits_yaw, logits_x, logits_y, value = self.heads(enc, mask=mask, gating_lambda=self.gating_lambda)
        return logits_pick, logits_yaw, logits_x, logits_y, value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=os.path.join(cfg["train"]["run_dir"], time.strftime("%Y%m%d-%H%M%S")))

    vecenv = make_vec_env(cfg)
    L, W = cfg["env"]["grid"]
    in_ch = 8 + 5
    model = ActorCritic(cfg, in_ch=in_ch).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg["ppo"]["lr"])

    storage = RolloutBuffer(cfg["ppo"]["rollout_length"], cfg["env"]["num_envs"], (in_ch, L, W), device)

    obs = torch.zeros((cfg["env"]["num_envs"], in_ch, L, W), dtype=torch.float32, device=device)
    global_steps = 0
    while global_steps < cfg["train"]["total_steps"]:
        storage.reset()
        for t in range(cfg["ppo"]["rollout_length"]):
            logits_pick, logits_yaw, logits_x, logits_y, value = model.forward_policy(obs)

            dist_p = Categorical(logits=logits_pick)
            dist_yaw = Categorical(logits=logits_yaw)
            dist_x = Categorical(logits=logits_x)
            dist_y = Categorical(logits=logits_y)

            a_pick = dist_p.sample()
            a_yaw = dist_yaw.sample()
            a_x = dist_x.sample()
            a_y = dist_y.sample()
            logprob = dist_p.log_prob(a_pick) + dist_yaw.log_prob(a_yaw) + dist_x.log_prob(a_x) + dist_y.log_prob(a_y)

            # Pass actions to environment
            actions_np = torch.stack([a_pick, a_yaw, a_x, a_y], dim=1).cpu().numpy()
            next_obs_np, reward_np, done_np, infos = vecenv.step(actions_np)
            
            reward = torch.as_tensor(reward_np).to(device)
            done = torch.as_tensor(done_np).to(device)

            storage.add(obs, torch.stack([a_pick,a_yaw,a_x,a_y], dim=1), reward, done, value.detach(), logprob.detach())
            obs = torch.as_tensor(next_obs_np).to(device)

            global_steps += cfg["env"]["num_envs"]
            if global_steps % cfg["train"]["log_interval"] == 0:
                writer.add_scalar("train/reward_mean", reward.mean().item(), global_steps)

        ppo_update(model, optim, storage, cfg)

        if global_steps % cfg["train"]["ckpt_interval"] == 0:
            ckpt_path = os.path.join(cfg["train"]["run_dir"], f"ckpt_{global_steps}.pt")
            os.makedirs(cfg["train"]["run_dir"], exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)

    torch.save(model.state_dict(), os.path.join(cfg["train"]["run_dir"], "last.pt"))
    writer.close()

if __name__ == "__main__":
    main()
