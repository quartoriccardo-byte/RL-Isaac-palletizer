
import os, argparse
import numpy as np
import torch
from pallet_rl.algo.utils import load_config
from pallet_rl.envs.vec_env_setup import make_vec_env
from pallet_rl.models.encoder2d import Encoder2D
from pallet_rl.models.unet2d import UNet2D
from pallet_rl.models.policy_heads import PolicyHeads

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

    def forward_policy(self, obs, mask=None):
        enc = self.encoder(obs)
        # mask = self.unet(obs) # Ignored
        logits_pick, logits_yaw, logits_pos, value = self.heads(enc, mask=mask, gating_lambda=self.gating_lambda)
        return logits_pick, logits_yaw, logits_pos, value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vecenv = make_vec_env(cfg)
    L, W = cfg["env"]["grid"]
    in_ch = 8 + 5
    model = ActorCritic(cfg, in_ch=in_ch).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    obs = torch.zeros((cfg["env"]["num_envs"], in_ch, L, W), dtype=torch.float32, device=device)
    # Dummy evaluation loop
    for _ in range(10):
        with torch.no_grad():
            # No mask needed for dummy eval? Or get from vecenv?
            # Model definition expects `mask` optional.
            # Output unpacking: 4 values
            logits_pick, logits_yaw, logits_pos, value = model.forward_policy(obs)
            
            # Simple greedy or sample debug
            from torch.distributions.categorical import Categorical
            dist_p = Categorical(logits=logits_pick)
            dist_yaw = Categorical(logits=logits_yaw)
            dist_pos = Categorical(logits=logits_pos)
            
            a_pick = dist_p.sample()
            a_yaw = dist_yaw.sample()
            a_pos = dist_pos.sample()
            
            x = a_pos // W
            y = a_pos % W
            
            print(f"Sampled Action: Pick={a_pick.item()}, Yaw={a_yaw.item()}, X={x.item()}, Y={y.item()}")

        obs = torch.zeros_like(obs)
    print("Eval done (dummy). Integrate with Isaac env for real metrics.")

if __name__ == "__main__":
    main()
