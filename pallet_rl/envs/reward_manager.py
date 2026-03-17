"""
Reward computation for the PalletTask environment.

All reward terms are implemented as pure functions operating on tensors,
making them independently testable without simulation.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pallet_rl.envs.pallet_task import PalletTask


def compute_rewards(env: PalletTask) -> torch.Tensor:
    """
    Compute rewards (pure PyTorch, JIT-compatible).

    Includes:
      - Height constraint penalty (invalid placement attempt)
      - Settling stability rewards (drift, fall, stable)
      - Infeasible payload penalty
      - Buffer store/retrieve/age incentives
      - Volume bonus for successful placements
      - Settled KPI evaluation

    Returns:
        rewards: ``(N,)`` tensor
    """
    n = env.num_envs
    device = env._device
    cfg = env.cfg

    rewards = torch.zeros(n, device=device)

    # ------------------------------------------------------------------
    # Height-invalid penalty
    # ------------------------------------------------------------------
    rewards += cfg.reward_invalid_height * env._height_invalid_mask.float()
    # Track invalid action rate for diagnostics
    if env._height_invalid_mask.any():
        env._kpi_invalid_action_count = getattr(env, "_kpi_invalid_action_count", torch.zeros(1, device=device))
        env._kpi_invalid_action_count += env._height_invalid_mask.float().sum()

    # ------------------------------------------------------------------
    # Infeasible payload penalty
    # ------------------------------------------------------------------
    rewards += cfg.reward_infeasible * env._infeasible_mask.float()
    if env._infeasible_mask.any():
        env._kpi_infeasible_count += env._infeasible_mask.float().sum()

    # ------------------------------------------------------------------
    # Buffer incentives
    # ------------------------------------------------------------------
    rewards -= 0.1 * env.store_mask.float()
    rewards += 2.0 * env.valid_retrieve.float()
    ages = env.buffer_state[:, :, 4].sum(dim=1)
    rewards -= 0.01 * ages

    # ------------------------------------------------------------------
    # Placement success / failure
    # ------------------------------------------------------------------
    valid_eval = env.last_moved_box_id >= 0

    if "boxes" in env.scene.keys() and valid_eval.any():
        rewards = _eval_placement_rewards(env, rewards, valid_eval, n, device)

    # ------------------------------------------------------------------
    # Queue settled KPI evaluations
    # ------------------------------------------------------------------
    _queue_kpi_evaluations(env)
    _evaluate_settled_kpis(env)

    # ------------------------------------------------------------------
    # Settling stability evaluation
    # ------------------------------------------------------------------
    _evaluate_settling(env, rewards, n)

    # Payload accumulation for utilization metric
    env._kpi_total_payload += env.payload_kg.sum()

    # ------------------------------------------------------------------
    # Log task KPIs to extras
    # ------------------------------------------------------------------
    _log_kpis(env)

    return rewards


# =============================================================================
# Internal Helpers
# =============================================================================

def _eval_placement_rewards(
    env: PalletTask,
    rewards: torch.Tensor,
    valid_eval: torch.Tensor,
    n: int,
    device,
) -> torch.Tensor:
    """Evaluate immediate placement success/failure rewards."""
    from pallet_rl.utils.quaternions import quat_angle_deg

    cfg = env.cfg
    valid_envs = valid_eval.nonzero(as_tuple=False).flatten()
    eval_box_idx = env.last_moved_box_id[valid_envs]
    global_idx = valid_envs * cfg.max_boxes + eval_box_idx
    current_pos, current_quat = env._get_box_pos_quat(global_idx)

    target_pos = env.last_target_pos[valid_envs]
    target_quat = env.last_target_quat[valid_envs]

    dist = torch.norm(current_pos[:, :2] - target_pos[:, :2], dim=-1)
    fell = current_pos[:, 2] < 0.05

    unstable_xy = dist > cfg.drift_xy_threshold
    rot_error_deg = quat_angle_deg(current_quat, target_quat)
    unstable_rot = rot_error_deg > cfg.drift_rot_threshold
    unstable = unstable_xy | unstable_rot

    rot_only_unstable = unstable_rot & ~unstable_xy & ~fell
    env._kpi_unstable_rot_count += rot_only_unstable.float().sum()

    failure_valid = fell | unstable
    success_valid = ~failure_valid

    failure = torch.zeros(n, dtype=torch.bool, device=device)
    success = torch.zeros(n, dtype=torch.bool, device=device)
    failure[valid_envs] = failure_valid
    success[valid_envs] = success_valid

    failure = failure & env.active_motion_mask
    success = success & env.active_motion_mask

    rewards -= 10.0 * failure.float()
    rewards += 1.0 * success.float()

    # Volume bonus
    dims = env.box_dims[valid_envs, eval_box_idx]
    vol = dims[:, 0] * dims[:, 1] * dims[:, 2]
    vol_rewards = torch.zeros(n, device=device)
    vol_rewards[valid_envs] = vol * success_valid.float()
    rewards += vol_rewards

    return rewards


def _queue_kpi_evaluations(env: PalletTask):
    """Queue place/retrieve actions for settled KPI evaluation."""
    cfg = env.cfg

    valid_place = env.active_place_mask & ~env._height_invalid_mask & (env.last_moved_box_id >= 0)
    if valid_place.any():
        place_envs = valid_place.nonzero(as_tuple=False).flatten()
        env._kpi_countdown[place_envs] = cfg.kpi_settle_steps + 1
        env._kpi_pending_type[place_envs] = 1
        env._kpi_pending_box_id[place_envs] = env.last_moved_box_id[place_envs]
        env._kpi_pending_target[place_envs] = env.last_target_pos[place_envs]
        env._kpi_pending_target_quat[place_envs] = env.last_target_quat[place_envs]

    if env.valid_retrieve.any():
        retr_envs = env.valid_retrieve.nonzero(as_tuple=False).flatten()
        env._kpi_countdown[retr_envs] = cfg.kpi_settle_steps + 1
        env._kpi_pending_type[retr_envs] = 2
        env._kpi_pending_box_id[retr_envs] = env.last_moved_box_id[retr_envs]
        env._kpi_pending_target[retr_envs] = env.last_target_pos[retr_envs]
        env._kpi_pending_target_quat[retr_envs] = env.last_target_quat[retr_envs]


def _evaluate_settled_kpis(env: PalletTask):
    """Evaluate KPIs for completed settling windows."""
    from pallet_rl.utils.quaternions import quat_angle_deg

    cfg = env.cfg
    pending = env._kpi_countdown > 0
    env._kpi_countdown[pending] -= 1

    ready_mask = (env._kpi_countdown == 0) & (env._kpi_pending_type > 0)
    if "boxes" not in env.scene.keys() or not ready_mask.any():
        return

    ready_envs = ready_mask.nonzero(as_tuple=False).flatten()
    eval_box_ids = env._kpi_pending_box_id[ready_envs]
    eval_targets = env._kpi_pending_target[ready_envs]
    eval_target_quats = env._kpi_pending_target_quat[ready_envs]
    eval_types = env._kpi_pending_type[ready_envs]

    global_idx = ready_envs * cfg.max_boxes + eval_box_ids
    settled_pos, settled_quat = env._get_box_pos_quat(global_idx)

    dist = torch.norm(settled_pos[:, :2] - eval_targets[:, :2], dim=-1)
    fell = settled_pos[:, 2] < 0.05
    unstable_xy = dist > cfg.drift_xy_threshold
    rot_error_deg = quat_angle_deg(settled_quat, eval_target_quats)
    unstable_rot = rot_error_deg > cfg.drift_rot_threshold
    unstable = unstable_xy | unstable_rot
    success = ~(fell | unstable)

    place_mask = eval_types == 1
    retr_mask = eval_types == 2

    env._kpi_place_success_count += (success & place_mask).float().sum()
    env._kpi_place_fail_count += (~success & place_mask).float().sum()
    env._kpi_retrieve_success_count += (success & retr_mask).float().sum()
    env._kpi_retrieve_fail_count += (~success & retr_mask).float().sum()
    env._kpi_eval_count += len(ready_envs)

    env._kpi_pending_type[ready_envs] = 0
    env._kpi_pending_box_id[ready_envs] = -1


def _evaluate_settling(env: PalletTask, rewards: torch.Tensor, n: int):
    """Evaluate settling stability and apply drift/fall/stable rewards."""
    cfg = env.cfg
    device = env._device

    settling = env._settle_countdown > 0
    env._settle_countdown[settling] -= 1

    # Falls during settling
    if "boxes" in env.scene.keys() and settling.any():
        settling_envs = settling.nonzero(as_tuple=False).flatten()
        box_ids = env._settle_box_id[settling_envs]
        valid_box_mask = box_ids >= 0
        if valid_box_mask.any():
            valid_settling_envs = settling_envs[valid_box_mask]
            valid_box_ids = box_ids[valid_box_mask]
            global_idx = valid_settling_envs * cfg.max_boxes + valid_box_ids
            current_pos, _ = env._get_box_pos_quat(global_idx)
            fell = current_pos[:, 2] < 0.05
            if fell.any():
                collapse_envs = valid_settling_envs[fell]
                rewards[collapse_envs] += cfg.reward_fall
                env._settle_countdown[collapse_envs] = 0
                env._settle_box_id[collapse_envs] = -1
                env._kpi_collapse_count += fell.float().sum()

    # Completed settling evaluation
    done_settling = (env._settle_countdown == 0) & (env._settle_box_id >= 0)
    if "boxes" in env.scene.keys() and done_settling.any():
        done_envs = done_settling.nonzero(as_tuple=False).flatten()
        box_ids = env._settle_box_id[done_envs]
        global_idx = done_envs * cfg.max_boxes + box_ids

        current_pos, current_quat = env._get_box_pos_quat(global_idx)
        target_pos = env._settle_target_pos[done_envs]
        target_quat = env._settle_target_quat[done_envs]

        drift_xy = torch.norm(current_pos[:, :2] - target_pos[:, :2], dim=-1)
        quat_dot = (current_quat * target_quat).sum(dim=-1).abs()
        drift_rot_rad = 2 * torch.acos(quat_dot.clamp(-1 + 1e-7, 1 - 1e-7))
        drift_rot_deg = drift_rot_rad * 180.0 / 3.14159265359

        fell = current_pos[:, 2] < 0.05
        exceeded_drift = (drift_xy > cfg.drift_xy_threshold) | (drift_rot_deg > cfg.drift_rot_threshold)

        fell_envs = done_envs[fell]
        if len(fell_envs) > 0:
            rewards[fell_envs] += cfg.reward_fall
            env._kpi_collapse_count += fell.float().sum()

        drifted_mask = exceeded_drift & ~fell
        drifted_envs = done_envs[drifted_mask]
        if len(drifted_envs) > 0:
            rewards[drifted_envs] += cfg.reward_drift
            env._kpi_drift_count += drifted_mask.float().sum()

        stable_mask = ~exceeded_drift & ~fell
        stable_envs = done_envs[stable_mask]
        if len(stable_envs) > 0:
            rewards[stable_envs] += cfg.reward_stable
            env._kpi_stable_count += stable_mask.float().sum()

        env._kpi_total_drift_xy += drift_xy.sum()
        env._kpi_total_drift_deg += drift_rot_deg.sum()
        env._kpi_settle_eval_count += len(done_envs)

        env._settle_box_id[done_envs] = -1


def _log_kpis(env: PalletTask):
    """Compute and log task KPIs to ``env.extras`` for TensorBoard."""
    total_place = env._kpi_place_success_count + env._kpi_place_fail_count
    total_retr = env._kpi_retrieve_success_count + env._kpi_retrieve_fail_count
    total_settle = env._kpi_settle_eval_count + 1e-8

    place_success_rate = env._kpi_place_success_count / (total_place + 1e-8)
    place_failure_rate = env._kpi_place_fail_count / (total_place + 1e-8)
    retrieve_success_rate = env._kpi_retrieve_success_count / (total_retr + 1e-8)

    store_attempts = env.store_mask.float().sum()
    store_accept_rate = env.valid_store.float().sum() / (store_attempts + 1e-8)
    buffer_occupancy = env.buffer_has_box.float().mean()

    drift_rate = env._kpi_drift_count / total_settle
    collapse_rate = env._kpi_collapse_count / total_settle
    infeasible_rate = env._kpi_infeasible_count / (env._kpi_eval_count + 1e-8)
    avg_drift_xy = env._kpi_total_drift_xy / total_settle
    avg_drift_deg = env._kpi_total_drift_deg / total_settle
    payload_utilization = env.payload_kg.mean() / env.cfg.max_payload_kg
    unstable_rot_rate = env._kpi_unstable_rot_count / (env._kpi_eval_count + 1e-8)

    # Episode-level metrics (success / failure / end reasons)
    total_episodes = env._kpi_episode_count + 1e-8
    success_episode_rate = env._kpi_success_episodes / total_episodes
    infeasible_episode_rate = env._kpi_infeasible_episodes / total_episodes
    failure_episode_rate = env._kpi_failure_episodes / total_episodes
    buffer_nonempty_end_rate = env._kpi_buffer_nonempty_at_end / total_episodes

    invalid_action_rate = getattr(env, "_kpi_invalid_action_count", torch.zeros(1)).to(env._device) / (
        total_settle + 1e-8
    )

    env.extras["metrics/place_success_rate"] = place_success_rate.detach().cpu().item()
    env.extras["metrics/place_failure_rate"] = place_failure_rate.detach().cpu().item()
    env.extras["metrics/retrieve_success_rate"] = retrieve_success_rate.detach().cpu().item()
    env.extras["metrics/store_accept_rate"] = store_accept_rate.detach().cpu().item()
    env.extras["metrics/buffer_occupancy"] = buffer_occupancy.detach().cpu().item()
    env.extras["metrics/drift_rate"] = drift_rate.detach().cpu().item()
    env.extras["metrics/collapse_rate"] = collapse_rate.detach().cpu().item()
    env.extras["metrics/infeasible_rate"] = infeasible_rate.detach().cpu().item()
    env.extras["metrics/avg_drift_xy"] = avg_drift_xy.detach().cpu().item()
    env.extras["metrics/avg_drift_deg"] = avg_drift_deg.detach().cpu().item()
    env.extras["metrics/payload_utilization"] = payload_utilization.detach().cpu().item()
    env.extras["metrics/unstable_rot_rate"] = unstable_rot_rate.detach().cpu().item()
    env.extras["metrics/episode_success_rate"] = success_episode_rate.detach().cpu().item()
    env.extras["metrics/episode_infeasible_rate"] = infeasible_episode_rate.detach().cpu().item()
    env.extras["metrics/episode_failure_rate"] = failure_episode_rate.detach().cpu().item()
    env.extras["metrics/episode_buffer_nonempty_end_rate"] = buffer_nonempty_end_rate.detach().cpu().item()
    env.extras["metrics/invalid_action_rate"] = invalid_action_rate.detach().cpu().item()
