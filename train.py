
import os, argparse, time
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from pallet_rl.algo.utils import load_config
from pallet_rl.envs.vec_env_setup import make_vec_env
from pallet_rl.models.encoder2d import Encoder2D
from pallet_rl.models.unet2d import UNet2D
from pallet_rl.models.policy_heads import SpatialPolicyHead
from pallet_rl.algo.storage import RolloutBuffer
from pallet_rl.algo.ppo import ppo_update

class ActorCritic(torch.nn.Module):
    def __init__(self, cfg, in_ch:int):
        super().__init__()
        L, W = cfg["env"]["grid"]
        base = cfg["model"]["encoder2d"]["base_channels"]
        self.encoder = Encoder2D(in_channels=in_ch, features=base)
        self.unet = UNet2D(in_ch=in_ch, base=cfg["model"]["unet2d"]["base_channels"])
        self.heads = SpatialPolicyHead(self.encoder.out_ch, (L, W),
                                 n_pick=cfg["env"]["buffer_N"],
                                 n_yaw=len(cfg["env"]["yaw_orients"]),
                                 hidden=cfg["model"]["policy_heads"]["hidden"])
        self.gating_lambda = cfg["mask"]["gating_lambda"]

    def forward_policy(self, obs, mask=None):
        enc = self.encoder(obs)
        outputs = self.heads(enc, mask=mask, gating_lambda=self.gating_lambda)
        return outputs

def main():
    use_isaac = os.environ.get("USE_ISAACLAB", "0") == "1"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--headless", action="store_true", default=False)
    
    # Isaac Lab Imports
    if use_isaac:
        try:
            from omni.isaac.lab.app import AppLauncher
            AppLauncher.add_app_launcher_args(parser)
        except ImportError:
            import sys
            if "--headless" in sys.argv or "-h" in sys.argv: # Simple check
                 pass # ArgumentParser will handle it or we check manually
            
            # Robust check for headless from sys.argv because argparse isn't parsed yet
            is_headless = "--headless" in sys.argv
            if is_headless:
                print("FATAL ERROR: '--headless' requested but Isaac Lab libraries ('omni') are missing.")
                print("Cannot run headless with DummyVecEnv. Please verify your environment variables and python path.")
                sys.exit(1)
            else:
                print("WARNING: Failed to import 'omni'. Isaac Lab libraries are missing. Falling back to DummyVecEnv.")
                use_isaac = False
                os.environ["USE_ISAACLAB"] = "0"
        
    args = parser.parse_args()
    
    simulation_app = None
    if use_isaac:
        from omni.isaac.lab.app import AppLauncher
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app
        
    try:
        cfg = load_config(args.config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        writer = SummaryWriter(log_dir=os.path.join(cfg["train"]["run_dir"], time.strftime("%Y%m%d-%H%M%S")))
        
        vecenv = make_vec_env(cfg)
        L, W = cfg["env"]["grid"]
        in_ch = 8 + 5
        model = ActorCritic(cfg, in_ch=in_ch).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=cfg["ppo"]["lr"])

        storage = RolloutBuffer(cfg["ppo"]["rollout_length"], cfg["env"]["num_envs"], (in_ch, L, W), device)
        storage.init_masks((L*W,)) # Initialize masks storage

        obs = torch.zeros((cfg["env"]["num_envs"], in_ch, L, W), dtype=torch.float32, device=device)
        
        # Episode tracking
        ep_lens = torch.zeros(cfg["env"]["num_envs"], device=device)
        ep_rets = torch.zeros(cfg["env"]["num_envs"], device=device)
        
        global_steps = 0
        while global_steps < cfg["train"]["total_steps"]:
            storage.reset()
            for t in range(cfg["ppo"]["rollout_length"]):
                # Get action mask
                mask_np = vecenv.get_action_mask()
                mask = torch.as_tensor(mask_np, device=device)

                logits_pick, logits_yaw, logits_pos, value = model.forward_policy(obs, mask=mask)

                dist_p = Categorical(logits=logits_pick)
                dist_yaw = Categorical(logits=logits_yaw)
                dist_pos = Categorical(logits=logits_pos)

                a_pick = dist_p.sample()
                a_yaw = dist_yaw.sample()
                a_pos = dist_pos.sample()
                
                # Decode pos
                a_x = a_pos // W
                a_y = a_pos % W
                
                logprob = dist_p.log_prob(a_pick) + dist_yaw.log_prob(a_yaw) + dist_pos.log_prob(a_pos)

                # Pass actions to environment
                actions_np = torch.stack([a_pick, a_yaw, a_x, a_y], dim=1).cpu().numpy()
                next_obs_np, reward_np, done_np, infos = vecenv.step(actions_np)
                
                reward = torch.as_tensor(reward_np).to(device)
                done = torch.as_tensor(done_np).to(device)

                # Track episodes
                ep_lens += 1
                ep_rets += reward
                
                # Log completed episodes
                finished = done.nonzero(as_tuple=False).squeeze(-1)
                if len(finished) > 0:
                    avg_len = ep_lens[finished].mean().item()
                    avg_ret = ep_rets[finished].mean().item()
                    writer.add_scalar("train/episode_length", avg_len, global_steps)
                    writer.add_scalar("train/reward_mean", avg_ret, global_steps)
                    
                    # Reset trackers
                    ep_lens[finished] = 0
                    ep_rets[finished] = 0

                storage.add(obs, torch.stack([a_pick,a_yaw,a_x,a_y], dim=1), reward, done, value.detach(), logprob.detach(), mask=mask)
                obs = torch.as_tensor(next_obs_np).to(device)

                global_steps += cfg["env"]["num_envs"]
                if global_steps % cfg["train"]["log_interval"] == 0:
                    # Log KPIs
                    vol_eff = np.mean([info.get("volume_ratio", 0.0) for info in infos])
                    stab_rate = np.mean([float(info.get("stable", 0.0)) for info in infos])
                    writer.add_scalar("train/volume_efficiency", vol_eff, global_steps)
                    writer.add_scalar("train/stability_rate", stab_rate, global_steps)

            loss_metrics = ppo_update(model, optim, storage, cfg)
            # Log PPO metrics
            if loss_metrics is not None:
                for k, v in loss_metrics.items():
                    writer.add_scalar(k, v, global_steps)

            if global_steps % cfg["train"]["ckpt_interval"] == 0:
                ckpt_path = os.path.join(cfg["train"]["run_dir"], f"ckpt_{global_steps}.pt")
                os.makedirs(cfg["train"]["run_dir"], exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)

        torch.save(model.state_dict(), os.path.join(cfg["train"]["run_dir"], "last.pt"))
        writer.close()
        
    finally:
        if simulation_app is not None:
            simulation_app.close()

if __name__ == "__main__":
    main()
