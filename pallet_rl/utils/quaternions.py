"""
Quaternion convention helpers.

We standardize the following conventions:
- Isaac Lab scene APIs expose quaternions as (w, x, y, z)
- NVIDIA Warp expects quaternions as (x, y, z, w)

Internal rule:
- All interaction with Isaac Lab (scene writes/reads) uses (w, x, y, z).
- All interaction with Warp uses (x, y, z, w).

These helpers perform explicit, shape-checked conversions at the boundaries.
"""

from __future__ import annotations

from typing import Tuple

import torch


def _assert_quat_last_dim(q: torch.Tensor) -> None:
    """Ensure tensor has a last dimension of size 4."""
    if q.shape[-1] != 4:
        raise AssertionError(f"Expected quaternion with last dim=4, got shape {tuple(q.shape)}")


def wxyz_to_xyzw(q_wxyz: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion from (w, x, y, z) to (x, y, z, w).

    Args:
        q_wxyz: Tensor [..., 4] ordered as (w, x, y, z).

    Returns:
        Tensor [..., 4] ordered as (x, y, z, w).
    """
    _assert_quat_last_dim(q_wxyz)
    w = q_wxyz[..., 0:1]
    xyz = q_wxyz[..., 1:4]
    return torch.cat([xyz, w], dim=-1)


def xyzw_to_wxyz(q_xyzw: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion from (x, y, z, w) to (w, x, y, z).

    Args:
        q_xyzw: Tensor [..., 4] ordered as (x, y, z, w).

    Returns:
        Tensor [..., 4] ordered as (w, x, y, z).
    """
    _assert_quat_last_dim(q_xyzw)
    xyz = q_xyzw[..., 0:3]
    w = q_xyzw[..., 3:4]
    return torch.cat([w, xyz], dim=-1)


def is_unit_quaternion(q: torch.Tensor, atol: float = 1e-4) -> torch.Tensor:
    """
    Check (per element) whether quaternions are approximately unit length.

    Args:
        q: Tensor [..., 4]
        atol: Absolute tolerance on |q| - 1.

    Returns:
        Bool tensor [...], True where |q| is ~1.
    """
    _assert_quat_last_dim(q)
    norm = torch.linalg.norm(q, dim=-1)
    return torch.isclose(norm, torch.ones_like(norm), atol=atol)


def quat_angle_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Compute the angular distance (in degrees) between two quaternions.
    
    Uses the formula: angle = 2 * acos(|q1 · q2|)
    The absolute value handles the quaternion double-cover (q == -q).
    
    Args:
        q1: Tensor [..., 4] first quaternion (any convention, must match q2).
        q2: Tensor [..., 4] second quaternion (same convention as q1).
    
    Returns:
        Tensor [...] angular distance in degrees.
    """
    _assert_quat_last_dim(q1)
    _assert_quat_last_dim(q2)
    # Dot product, take abs to handle sign ambiguity
    dot = (q1 * q2).sum(dim=-1).abs()
    # Clamp for numerical stability before acos
    dot_clamped = dot.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle_rad = 2.0 * torch.acos(dot_clamped)
    angle_deg = angle_rad * (180.0 / 3.14159265358979323846)
    return angle_deg


__all__: Tuple[str, ...] = ("wxyz_to_xyzw", "xyzw_to_wxyz", "is_unit_quaternion", "quat_angle_deg")

