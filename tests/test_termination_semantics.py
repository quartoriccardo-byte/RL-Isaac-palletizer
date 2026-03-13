"""
Tests for high-level termination semantics (success vs. failure).

These tests are Isaac-free and operate on simple tensors that mirror
the logic in :meth:`PalletTask._get_dones`. They validate that:

  - Success is based on ``num_boxes`` (not ``max_boxes``).
  - Success cannot occur while the buffer is non-empty.
  - Success is blocked while a settling window is still pending.
"""

from __future__ import annotations

import torch


def termination_core(
    box_idx: torch.Tensor,
    num_boxes: int,
    buffer_has_box: torch.Tensor,
    infeasible_mask: torch.Tensor,
    settle_box_id: torch.Tensor,
) -> torch.Tensor:
    """
    Minimal re-implementation of the success branch from
    :meth:`PalletTask._get_dones` for Isaac-free testing.

    Returns:
        success_only: (N,) bool tensor indicating successful completion.
    """
    all_fresh_consumed = box_idx >= num_boxes
    buffer_nonempty = buffer_has_box.any(dim=1)
    no_pending_settle = settle_box_id < 0

    success_mask = all_fresh_consumed & (~buffer_nonempty) & no_pending_settle
    terminated = infeasible_mask.clone()
    success_only = success_mask & ~terminated
    return success_only


def test_success_requires_num_boxes_and_empty_buffer():
    """Success uses num_boxes and requires buffer to be empty."""
    device = torch.device("cpu")
    n = 3
    num_boxes = 5

    box_idx = torch.tensor([5, 5, 4], dtype=torch.long, device=device)
    buffer_has_box = torch.zeros(n, 2, dtype=torch.bool, device=device)
    buffer_has_box[1, 0] = True  # env 1 still has buffered box

    infeasible = torch.zeros(n, dtype=torch.bool, device=device)
    settle_box_id = torch.full((n,), -1, dtype=torch.long, device=device)

    success_only = termination_core(box_idx, num_boxes, buffer_has_box, infeasible, settle_box_id)

    # env 0: box_idx == num_boxes, buffer empty  -> success
    # env 1: box_idx == num_boxes, buffer nonempty -> NOT success
    # env 2: box_idx < num_boxes -> NOT success
    expected = torch.tensor([True, False, False], dtype=torch.bool, device=device)
    assert torch.equal(success_only, expected), f"{success_only} != {expected}"


def test_success_blocked_while_settling_pending():
    """Success must wait for the last placement to finish settling."""
    device = torch.device("cpu")
    n = 2
    num_boxes = 3

    box_idx = torch.tensor([3, 3], dtype=torch.long, device=device)
    buffer_has_box = torch.zeros(n, 2, dtype=torch.bool, device=device)
    infeasible = torch.zeros(n, dtype=torch.bool, device=device)

    # env 0: still settling last box (id=2)
    # env 1: no pending settle
    settle_box_id = torch.tensor([2, -1], dtype=torch.long, device=device)

    success_only = termination_core(box_idx, num_boxes, buffer_has_box, infeasible, settle_box_id)

    expected = torch.tensor([False, True], dtype=torch.bool, device=device)
    assert torch.equal(success_only, expected), f"{success_only} != {expected}"

