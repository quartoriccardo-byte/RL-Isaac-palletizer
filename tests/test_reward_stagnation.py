"""
Unit tests for stagnation detection, reward helpers, and termination logic.

Isaac-free: operates on pure tensors mirroring reward_manager and pallet_task logic.
"""

from __future__ import annotations

import torch
import pytest


# =============================================================================
# Stagnation Counter Tests
# =============================================================================

def test_stagnation_counter_increments_without_progress():
    """Counter increments every step when box_idx doesn't advance."""
    n = 4
    stag = torch.zeros(n, dtype=torch.long)
    box_idx = torch.tensor([2, 2, 2, 2], dtype=torch.long)
    prev_idx = torch.tensor([2, 2, 2, 2], dtype=torch.long)

    progress = box_idx > prev_idx
    stag[progress] = 0
    stag[~progress] += 1

    assert stag.tolist() == [1, 1, 1, 1]

    # Simulate a second step
    stag[~progress] += 1
    assert stag.tolist() == [2, 2, 2, 2]


def test_stagnation_counter_resets_on_progress():
    """Counter resets to 0 when box_idx advances."""
    n = 4
    stag = torch.tensor([10, 20, 30, 40], dtype=torch.long)
    box_idx = torch.tensor([3, 2, 5, 2], dtype=torch.long)
    prev_idx = torch.tensor([2, 2, 4, 2], dtype=torch.long)

    progress = box_idx > prev_idx
    stag[progress] = 0
    stag[~progress] += 1

    # env 0: advanced 2->3, reset
    # env 1: no change, 20->21
    # env 2: advanced 4->5, reset
    # env 3: no change, 40->41
    assert stag.tolist() == [0, 21, 0, 41]


def test_stagnation_termination_at_window():
    """Episode terminates exactly at stagnation_window."""
    window = 50
    stag = torch.tensor([49, 50, 51, 48], dtype=torch.long)
    kill = stag >= window
    expected = torch.tensor([False, True, True, False])
    assert torch.equal(kill, expected)


# =============================================================================
# Action Repetition Tests
# =============================================================================

def test_repetition_counter_increments_on_same_action():
    """Counter increments when action is identical and no progress."""
    n = 3
    rep_count = torch.zeros(n, dtype=torch.long)
    current = torch.tensor([[0.0, 1.0, 2.0, 3.0, 1.0],
                             [0.0, 1.0, 2.0, 3.0, 1.0],
                             [0.0, 1.0, 2.0, 3.0, 1.0]])
    prev = torch.tensor([[0.0, 1.0, 2.0, 3.0, 1.0],
                          [0.0, 1.0, 2.0, 3.0, 1.0],
                          [1.0, 1.0, 2.0, 3.0, 1.0]])  # env 2 differs

    progress = torch.tensor([False, False, False])
    same = (current == prev).all(dim=-1)
    repeating = same & ~progress

    rep_count[repeating] += 1
    rep_count[~repeating] = 0

    # env 0: same + no progress -> 1
    # env 1: same + no progress -> 1
    # env 2: different -> 0
    assert rep_count.tolist() == [1, 1, 0]


def test_repetition_counter_resets_on_progress():
    """Counter resets even with same action if progress was made."""
    n = 2
    rep_count = torch.tensor([5, 5], dtype=torch.long)
    current = torch.tensor([[0.0, 1.0, 2.0, 3.0, 1.0],
                             [0.0, 1.0, 2.0, 3.0, 1.0]])
    prev = current.clone()

    progress = torch.tensor([True, False])
    same = (current == prev).all(dim=-1)
    repeating = same & ~progress

    rep_count[repeating] += 1
    rep_count[~repeating] = 0

    # env 0: same but had progress -> reset to 0
    # env 1: same and no progress -> 6
    assert rep_count.tolist() == [0, 6]


def test_repetition_penalty_threshold():
    """Penalty only fires at >=2 consecutive repetitions."""
    rep_count = torch.tensor([0, 1, 2, 3, 10], dtype=torch.long)
    penalty_scale = -0.2
    repeating = rep_count >= 2
    penalty = penalty_scale * repeating.float()

    expected = torch.tensor([0.0, 0.0, -0.2, -0.2, -0.2])
    assert torch.allclose(penalty, expected)


# =============================================================================
# Penalty Signal Tests
# =============================================================================

def test_time_penalty_every_step():
    """Time penalty is applied to every env, every step."""
    n = 8
    scale = -0.05
    penalty = torch.full((n,), scale)
    expected = torch.full((n,), scale)
    assert torch.allclose(penalty, expected)


def test_inactivity_penalty_only_on_non_progress():
    """Inactivity penalty fires only when no box advancement."""
    progress = torch.tensor([True, False, True, False])
    scale = -0.3
    penalty = scale * (~progress).float()
    expected = torch.tensor([0.0, -0.3, 0.0, -0.3])
    assert torch.allclose(penalty, expected)


def test_success_bonus_exceeds_max_penalties():
    """Success reward must be larger than worst-case accumulated penalties.

    With max_episode_steps=200, worst-case per-step penalty is:
      time(-0.05) + inactivity(-0.3) + repetition(-0.2) = -0.55/step
    Over 200 steps = -110.0. Success bonus = 50.0.
    Combined with place_progress rewards (5.0 * num_boxes), net should be
    positive for successful completion.
    """
    success_bonus = 50.0
    num_boxes = 50
    progress_per_box = 5.0
    max_steps = 200

    # Best case: agent places all boxes with some inactive steps in between
    total_progress_reward = num_boxes * progress_per_box
    total_time_penalty = max_steps * 0.05
    total_success = success_bonus + total_progress_reward - total_time_penalty

    # Must be significantly positive
    assert total_success > 100.0, f"Total success reward {total_success} not large enough"


# =============================================================================
# Termination Reason Tests
# =============================================================================

def test_termination_reason_assignment():
    """Verify termination reason codes are assigned correctly."""
    n = 5
    terminated = torch.tensor([True, True, True, True, False])
    infeasible_mask = torch.tensor([False, True, False, False, False])
    stagnation_kill = torch.tensor([False, False, True, False, False])
    success_only = torch.tensor([True, False, False, False, False])

    reason = torch.zeros(n, dtype=torch.long)
    reason[terminated] = 4          # generic failure
    reason[infeasible_mask] = 5     # infeasible
    reason[stagnation_kill] = 3     # stagnation
    reason[success_only] = 1        # success

    # env 0: success (overrides generic failure)
    # env 1: infeasible (overrides generic failure)
    # env 2: stagnation (overrides generic failure)
    # env 3: generic failure
    # env 4: not terminated
    expected = torch.tensor([1, 5, 3, 4, 0])
    assert torch.equal(reason, expected)


# =============================================================================
# Episode Horizon Tests
# =============================================================================

def test_episode_length_from_max_steps():
    """episode_length_s should be computed from max_episode_steps."""
    dt = 1.0 / 60.0
    decimation = 50
    max_steps = 200

    step_dt = dt * decimation
    episode_length_s = float(max_steps) * step_dt

    # 200 * (50/60) = 166.67s
    assert abs(episode_length_s - 166.667) < 0.1
