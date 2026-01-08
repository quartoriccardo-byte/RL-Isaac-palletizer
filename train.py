
import os, argparse, time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from pallet_rl.algo.utils import load_config
from pallet_rl.envs.vec_env_setup import make_vec_env
from pallet_rl.algo.ppo import PPO

def main():
    # Isaac Lab Check
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
            if "--headless" in sys.argv or "-h" in sys.argv:
                 pass
            is_headless = "--headless" in sys.argv
            if is_headless:
                print("FATAL ERROR: '--headless' requested but Isaac Lab libraries ('omni') are missing.")
                sys.exit(1)
            else:
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
        
        # Logging
        run_dir = f"runs/{cfg['env']['env_name']}_seed{cfg['seed']}_{time.strftime('%Y%m%d-%H%M%S')}"
        writer = SummaryWriter(log_dir=run_dir)
        
        # Environment
        vecenv = make_vec_env(cfg)
        
        # Agent
        obs_shape = vecenv.obs_shape # Should be available from vecenv
        # If not available on dummy, hardcode check
        if not hasattr(vecenv, 'obs_shape'):
             # Fallback
             L, W = 40, 48 # default?
             obs_shape = (13, L, W) 
             
        agent = PPO(cfg, obs_shape)
        
        # Storage
        num_steps = cfg["algo"]["n_steps"]
        num_envs = cfg["env"]["num_envs"]
        device = torch.device(cfg["device"])
        
        obs = torch.zeros((num_steps, num_envs, *obs_shape), device=device)
        actions = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device) # Flattened actions
        rewards = torch.zeros((num_steps, num_envs), device=device)
        dones = torch.zeros((num_steps, num_envs), device=device)
        logprobs = torch.zeros((num_steps, num_envs), device=device)
        values = torch.zeros((num_steps, num_envs), device=device)
        masks = torch.zeros((num_steps, num_envs, *obs_shape[1:]), dtype=torch.bool, device=device) # Approx shape
        
        # Reset env
        next_obs_np = vecenv.reset()
        next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
        
        global_step = 0
        update = 0
        total_updates = 1000 # Just a loop limit or infinite? "Run the main training loop"
        
        print("Starting training loop...")
        
        for update in range(1, total_updates+1):
            # Anneal learning rate? Prompt doesn't specify.
            
            # Rollout
            for step in range(num_steps):
                global_step += num_envs
                obs[step] = next_obs
                
                # Get Mask
                mask_np = vecenv.get_action_mask()
                mask = torch.as_tensor(mask_np, device=device)
                masks[step] = mask.view(num_envs, *obs_shape[1:]) # Ensure shape match if flatter
                
                with torch.no_grad():
                     action, logprob, value = agent.act(next_obs, mask=mask)
                     values[step] = value.flatten()
                
                actions[step] = action
                logprobs[step] = logprob
                
                # Execute action
                # `action` is a flat index tensor (Batch,)
                # vecenv.step decodes it internally
                next_obs_np, reward_np, done_np, infos = vecenv.step(action)
                
                rewards[step] = torch.tensor(reward_np, device=device).view(-1)
                next_done = torch.as_tensor(done_np, device=device).float()
                dones[step] = next_done
                
                next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
                
                # Logging episode info
                for info in infos:
                     if "episode" in info:
                         print(f"Episode Reward: {info['episode']['r']}")
                         writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)

            # Bootstrap value
            with torch.no_grad():
                 _, _, next_value = agent.act(next_obs, mask=torch.as_tensor(vecenv.get_action_mask(), device=device))
                 next_value = next_value.reshape(1, -1)
                 
            # GAE Calculation
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    nextvalues = values[t+1]
                
                delta = rewards[t] + cfg["algo"]["gamma"] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + cfg["algo"]["gamma"] * cfg["algo"]["gae_lambda"] * nextnonterminal * lastgaelam
            
            returns = advantages + values
            
            # Flatten batch
            b_obs = obs.reshape((-1,) + obs_shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_masks = masks.reshape((-1,) + masks.shape[2:])
            
            # Optimize
            loss = agent.update({
                "obs": b_obs,
                "actions": b_actions,
                "logprobs": b_logprobs,
                "advantages": b_advantages,
                "returns": b_returns,
                "masks": b_masks
            })
            
            writer.add_scalar("losses/policy_loss", loss, global_step)
            print(f"Update {update}, Loss: {loss}")
            
            # Save Checkpoint
            if update % 10 == 0:
                 torch.save(agent.policy.state_dict(), f"{run_dir}/agent_{update}.pt")

        writer.close()
        
    finally:
        if simulation_app is not None:
            simulation_app.close()

if __name__ == "__main__":
    main()
