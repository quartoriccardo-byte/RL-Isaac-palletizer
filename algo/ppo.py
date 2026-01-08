
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

def ppo_update(model, optimizer, storage, cfg):
    gamma = cfg["ppo"]["gamma"]; lam = cfg["ppo"]["gae_lambda"]
    clip_coef = cfg["ppo"]["clip_coef"]; ent_coef = cfg["ppo"]["ent_coef"]; vf_coef = cfg["ppo"]["vf_coef"]
    epochs = cfg["ppo"]["epochs_per_update"]; minibatches = cfg["ppo"]["minibatches_per_update"]

    T, N = storage.rewards.shape
    device = storage.rewards.device

    returns = torch.zeros((T, N), device=device)
    adv = torch.zeros((T, N), device=device)
    next_value = torch.zeros((N,), device=device)
    gae = torch.zeros((N,), device=device)
    for t in reversed(range(T)):
        delta = storage.rewards[t] + gamma * next_value * (~storage.dones[t]) - storage.values[t]
        gae = delta + gamma * lam * gae * (~storage.dones[t])
        adv[t] = gae
        returns[t] = gae + storage.values[t]
        next_value = storage.values[t]

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    B = T * N
    idx = torch.randperm(B, device=device)
    obs = storage.obs.reshape(B, *storage.obs.shape[2:])
    acts = storage.actions.reshape(B, 4)
    rets = returns.reshape(B)
    advs = adv.reshape(B)
    old_logp = storage.logprobs.reshape(B)
    masks = storage.masks.reshape(B, *storage.masks.shape[2:])

    batch_size = max(1, B // minibatches)

    model.train()
    for _ in range(epochs):
        for i in range(minibatches):
            j = idx[i*batch_size:(i+1)*batch_size]
            o = obs[j]; a = acts[j]; R = rets[j]; A = advs[j]; old_lp = old_logp[j]; m = masks[j]

            # Reconstruct joint position action from x, y
            # a is [pick, yaw, x, y]
            # Need W from config to compute pos index
            L, W = cfg["env"]["grid"]
            a_pos = a[:, 2] * W + a[:, 3]
            a_pos = a_pos.long()

            logits_pick, logits_yaw, logits_pos, value = model.forward_policy(o, mask=m)
            dist_p = Categorical(logits=logits_pick)
            dist_yaw = Categorical(logits=logits_yaw)
            dist_pos = Categorical(logits=logits_pos)

            lp = dist_p.log_prob(a[:,0]) + dist_yaw.log_prob(a[:,1]) + dist_pos.log_prob(a_pos)
            ratio = (lp - old_lp).exp()

            unclipped = ratio * A
            clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * A
            pg_loss = -torch.min(unclipped, clipped).mean()

            v_loss = 0.5 * F.mse_loss(value, R)
            ent = (dist_p.entropy() + dist_yaw.entropy() + dist_pos.entropy()).mean()

            loss = pg_loss + vf_coef * v_loss - ent_coef * ent

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
    return {
        "loss/policy": pg_loss.item(),
        "loss/value": v_loss.item(),
        "loss/entropy": ent.item()
    }
