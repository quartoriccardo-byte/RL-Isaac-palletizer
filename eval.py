
import os, argparse
import numpy as np
import torch
from pallet_rl.algo.utils import load_config
from pallet_rl.envs.vec_env_setup import make_vec_env
from pallet_rl.models.encoder2d import Encoder2D
from pallet_rl.models.unet2d import UNet2D
from pallet_rl.models.policy_heads import SpatialPolicyHead

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
    parser.add_argument("--checkpoint", type=str, required=True)
    
    # Isaac Lab Imports
    if use_isaac:
        from omni.isaac.lab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
        
    args = parser.parse_args()
    
    simulation_app = None
    if use_isaac:
        from omni.isaac.lab.app import AppLauncher
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app
    
    try:
        cfg = load_config(args.config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        vecenv = make_vec_env(cfg)
        L, W = cfg["env"]["grid"]
        in_ch = 8 + 5
        model = ActorCritic(cfg, in_ch=in_ch).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()

        obs_np = vecenv.reset()
        obs = torch.as_tensor(obs_np, device=device)
        
        print("Starting evaluation loop...")
        while True:
            mask_np = vecenv.get_action_mask()
            mask = torch.as_tensor(mask_np, device=device)

            with torch.no_grad():
                logits_pick, logits_yaw, logits_pos, value = model.forward_policy(obs, mask=mask)
                
                # Deterministic (Argmax) or Sample?
                # Usually eval is deterministic.
                a_pick = torch.argmax(logits_pick, dim=1)
                a_yaw = torch.argmax(logits_yaw, dim=1)
                a_pos = torch.argmax(logits_pos, dim=1)
                
                a_x = a_pos // W
                a_y = a_pos % W
                
                # Pass actions to environment
                actions_np = torch.stack([a_pick, a_yaw, a_x, a_y], dim=1).cpu().numpy()
                next_obs_np, reward_np, done_np, infos = vecenv.step(actions_np)
                
                obs = torch.as_tensor(next_obs_np).to(device)
                
            if use_isaac:
                # Assuming simulation_app handles stepping via vecenv.world.step() inside vecenv.step()
                # If we want to slow down for visualization?
                pass
                
    except KeyboardInterrupt:
        print("Evaluation stopped.")
    finally:
        if simulation_app is not None:
            simulation_app.close()

if __name__ == "__main__":
    main()
