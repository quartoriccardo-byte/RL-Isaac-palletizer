#!/usr/bin/env python3
"""
Physically plausible palletizing mockup video generator.

Two placement modes (--placement_mode):

  KINEMATIC (default) — guaranteed-visible, no PhysX failures:
    SPAWN → CARRY → LOWER → PLACE_KINEMATIC → PAUSE
    Boxes are moved to target pose and kept kinematic (no physics drop).
    No settle/validation/retry needed; boxes always appear.

  DROP — physics-based settling:
    SPAWN → CARRY → LOWER → SETTLE → PAUSE
    Boxes released near target, settle under gravity.
    Single-box retry on failure (no full-episode reset).

Key architecture:
  - Pallet PHYSICS is a simple cuboid collider (kinematic, stable).
  - Pallet STL is VISUAL ONLY, auto-aligned by pallet_task.py.
  - Visible floor slab comes from PalletTaskCfg.floor_visual_enabled.
  - displayColor + displayOpacity set on every box prim so they are
    visible even when HydraStorm ignores material bindings.
  - All pallet_task.py features (floor, mesh, mockup physics) are reused.

Run:
  ~/isaac-sim/python.sh scripts/mockup_video_physics.py \\
      --headless --output_path runs/mockup.mp4

  ~/isaac-sim/python.sh scripts/mockup_video_physics.py \\
      --headless --placement_mode drop --debug --output_path runs/drop.mp4

  ~/isaac-sim/python.sh scripts/mockup_video_physics.py \\
      --headless --output_path runs/heightmap.mp4 --record_mode heightmap

  ~/isaac-sim/python.sh scripts/mockup_video_physics.py \\
      --headless --output_path runs/both.mp4 --record_mode both

Three recording modes (--record_mode):

  rgb (default)     - Cinematic oblique RGB camera (unchanged behaviour).
  heightmap         - Agent's top-down heightmap (depth camera -> heightmap
                      in meters -> colormap). Noise disabled by default.
  both              - Side-by-side: RGB left, heightmap right.

CLI knobs:
  --record_mode, --hmap_vmin, --hmap_vmax, --hmap_colormap, --hmap_invert,
  --disable_depth_noise, --placement_mode, --debug, --carry_height,
  --lower_speed, --release_clearance, --settle_s, --freeze_after_settle,
  --max_retries, --num_boxes, --duration_s, etc.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import cv2

import torch
from pallet_rl.utils.device_utils import pick_supported_cuda_device


# ═══════════════════════════════════════════════════════════════════════
# 1) Parse CLI — MUST happen before AppLauncher / any isaaclab import
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Physically plausible palletiser mockup video"
    )
    # output / render
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--output_path", type=str, default="runs/mockup_physics.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration_s", type=float, default=20.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--enable_cameras", action="store_true", default=True)
    parser.add_argument("--cam_width", type=int, default=1280)
    parser.add_argument("--cam_height", type=int, default=720)

    # box counts & seed
    parser.add_argument("--num_boxes", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)

    # motion / stability knobs
    parser.add_argument("--carry_height", type=float, default=1.2,
                        help="Z height during carry phase (m)")
    parser.add_argument("--lower_speed", type=float, default=0.70,
                        help="Kinematic descent speed (m/s)")
    parser.add_argument("--release_clearance", type=float, default=0.02,
                        help="Height above target where box switches to dynamic (m)")
    parser.add_argument("--settle_s", type=float, default=3.0,
                        help="Max settle time per box (seconds)")
    parser.add_argument("--settle_vel_threshold", type=float, default=0.05,
                        help="Velocity norm below which box is considered settled")
    parser.add_argument("--freeze_after_settle", action="store_true", default=False,
                        help="Convert placed box to kinematic after settling "
                             "(opt-in; may break stacking contacts)")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Max retries per box before skipping")
    parser.add_argument("--sim_substeps", type=int, default=8,
                        help="Physics substeps per rendered frame")

    # pallet geometry
    parser.add_argument("--pallet_size_x", type=float, default=1.2)
    parser.add_argument("--pallet_size_y", type=float, default=0.8)
    parser.add_argument("--pallet_thickness", type=float, default=0.15)

    # placement mode
    parser.add_argument("--placement_mode", type=str, default="drop",
                        choices=["kinematic", "drop"],
                        help="'kinematic': boxes placed at target "
                             "pose, no physics drop. 'drop' (default): physics-based "
                             "settling after release.")

    # stacking guardrails
    parser.add_argument("--support_ratio_min", type=float, default=0.6,
                        help="Minimum area of support (0.0 to 1.0) required from the layer below")
    parser.add_argument("--edge_margin", type=float, default=0.02,
                        help="Allowed margin before box falls off the pallet edge")

    # debug diagnostics
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Print detailed diagnostic logs (prim paths, "
                             "poses, visibility attributes)")
    parser.add_argument("--debug_dump_frame", action="store_true", default=False,
                        help="Save one RGB frame as PNG after each successful placement")
    parser.add_argument("--debug_dump_path", type=str, default="runs/debug_frames",
                        help="Directory for debug frame PNGs")

    # ── recording mode ──────────────────────────────────────────────────
    parser.add_argument("--record_mode", type=str, default="rgb",
                        choices=["rgb", "heightmap", "both"],
                        help="What to record: 'rgb' (default, existing behaviour), "
                             "'heightmap' (agent top-down heightmap), "
                             "'both' (side-by-side RGB + heightmap)")

    # ── heightmap visualization ────────────────────────────────────────
    parser.add_argument("--hmap_vmin", type=float, default=0.0,
                        help="Heightmap visualization min clamp (meters)")
    parser.add_argument("--hmap_vmax", type=float, default=0.0,
                        help="Heightmap visualization max clamp (meters). "
                             "0 = auto from PalletTaskCfg.max_height")
    parser.add_argument("--hmap_colormap", type=str, default="inferno",
                        help="OpenCV colormap name for heightmap "
                             "(inferno, jet, turbo, viridis, magma, etc.)")
    parser.add_argument("--hmap_invert", action="store_true", default=False,
                        help="Invert colormap so high=dark")
    # FIX: use --enable_depth_noise (opt-in) instead of --disable_depth_noise
    # (store_true + default=True was impossible to toggle off)
    parser.add_argument("--enable_depth_noise", action="store_true",
                        default=False,
                        help="Enable depth sensor noise during heightmap "
                             "recording. By default noise is OFF for clean "
                             "video output. Pass this flag to simulate "
                             "realistic sensor noise.")

    # ── raw depth dump (works with --record_mode heightmap) ────────────
    parser.add_argument("--save_depth_raw", action="store_true",
                        help="Save raw depth frames as .npy")
    parser.add_argument("--depth_raw_dir", type=str, default="",
                        help="Output folder for raw depth; if empty, "
                             "use <output_path>_depth_raw")

    # ── debug: box sync verification ──────────────────────────────────
    parser.add_argument("--debug_box_sync", action="store_true", default=False,
                        help="After each placement, step once and print the "
                             "box world z to verify it left the parking pose")

    # extensions
    parser.add_argument("--exclude_isaaclab_tasks", action="store_true", default=True,
                        help="Exclude problematic isaaclab_tasks extension")

    # physics fallback & physx gpu buffer sizes
    parser.add_argument("--physics_device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device for physics simulation (cuda or cpu)")
    parser.add_argument("--physx_sync_launch", action="store_true", default=False,
                        help="Enable synchronous kernel launches for debugging")
    parser.add_argument("--gpu_found_lost_pairs_capacity", type=int, default=1048576,
                        help="PhysX GPU found/lost pairs capacity")
    parser.add_argument("--gpu_total_aggregate_pairs_capacity", type=int, default=1048576,
                        help="PhysX GPU total aggregate pairs capacity")
    parser.add_argument("--gpu_heap_capacity", type=int, default=67108864,
                        help="PhysX GPU heap capacity (bytes)")
    parser.add_argument("--gpu_temp_buffer_capacity", type=int, default=16777216,
                        help="PhysX GPU temp buffer capacity (bytes)")

    return parser.parse_known_args()


def inject_kit_args(args, unknown):
    """Inject safe Kit/Carb args BEFORE AppLauncher reads sys.argv."""
    if not args.enable_cameras:
        args.enable_cameras = True

    # ── GPU ordinal mapping for this machine ───────────────────────
    # /renderer/activeGpu  uses *Vulkan* ordinals → RTX 6000 = 2
    # /physics/cudaDevice   uses *CUDA*   ordinals → RTX 6000 = 0
    # These numbers intentionally differ; they are separate namespaces.
    vulkan_idx = "2"   # RTX 6000 in Vulkan device list
    cuda_idx   = "0"   # RTX 6000 in CUDA device list

    user_kit_args = [a for a in unknown if a.startswith("--/")]
    user_kit_paths = {a.split("=")[0] for a in user_kit_args}

    defaults = {
        "--/ngx/enabled": "--/ngx/enabled=false",
        "--/rtx/post/dlss/enabled": "--/rtx/post/dlss/enabled=false",
        "--/renderer/multiGpu/enabled": "--/renderer/multiGpu/enabled=false",
        "--/renderer/activeGpu": f"--/renderer/activeGpu={vulkan_idx}",
        "--/physics/cudaDevice": f"--/physics/cudaDevice={cuda_idx}",
    }
    
    if args.physx_sync_launch:
        defaults["--/physics/enableSynchronousKernelLaunches"] = "--/physics/enableSynchronousKernelLaunches=true"
    
    if args.exclude_isaaclab_tasks:
        # Check if the user already provided an extension exclusion override
        exclusion_idx = 0
        while any(p.startswith(f"--/app/extensions/excluded/{exclusion_idx}") for p in user_kit_paths):
            exclusion_idx += 1
        defaults[f"--/app/extensions/excluded/{exclusion_idx}"] = f"--/app/extensions/excluded/{exclusion_idx}='isaaclab_tasks'"
        print(f"[INFO] Excluding isaaclab_tasks extension at index {exclusion_idx}")

    for path, arg in defaults.items():
        if path not in user_kit_paths:
            sys.argv.append(arg)
    for arg in user_kit_args:
        if arg not in sys.argv:
            sys.argv.append(arg)


args, unknown = parse_args()

# ─── Force supported GPU (RTX 6000 vs 1080 Ti) ─────────────────────────
# pick_supported_cuda_device() selects the correct *CUDA* device for
# PyTorch tensors.  Kit settings (Vulkan + CUDA ordinals) are handled
# separately in inject_kit_args().
_cuda_idx, forced_device = pick_supported_cuda_device()
args.device = forced_device
print(f"[INFO] PyTorch CUDA device: {args.device}  (CUDA idx {_cuda_idx})")
print(f"[INFO] Kit renderer: Vulkan GPU 2 | PhysX CUDA device: 0")

inject_kit_args(args, unknown)


# ═══════════════════════════════════════════════════════════════════════
# 2) Launch Isaac Sim — BEFORE any isaaclab/pxr imports
# ═══════════════════════════════════════════════════════════════════════

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Force Kit settings AFTER AppLauncher, BEFORE env creation ──────────
# AppLauncher may overwrite /physics/cudaDevice during init.  We re-apply
# the correct values: renderer=Vulkan idx 2, PhysX=CUDA idx 0.
try:
    import carb.settings
    _s = carb.settings.get_settings()
    _s.set("/renderer/activeGpu", 2)       # Vulkan ordinal
    _s.set("/physics/cudaDevice", 0)       # CUDA ordinal
    _s.set("/renderer/multiGpu/enabled", False)
    print("[INFO] Post-launch Kit settings forced: "
          "activeGpu=2 (Vulkan), cudaDevice=0 (CUDA), multiGpu=false")
except Exception as _e:
    print(f"[WARN] Could not force post-launch Kit settings: {_e}")


# ═══════════════════════════════════════════════════════════════════════
# 3) Now safe to import Isaac Lab, pxr, torch, etc.
# ═══════════════════════════════════════════════════════════════════════

import numpy as np
import torch

from pxr import UsdPhysics, PhysxSchema, UsdShade, Sdf, Gf, UsdGeom

from pallet_rl.envs.pallet_task import PalletTask, PalletTaskCfg
from pallet_rl.utils.depth_heightmap import DepthHeightmapConverter, DepthHeightmapCfg


# ═══════════════════════════════════════════════════════════════════════
# Heightmap visualization helpers
# ═══════════════════════════════════════════════════════════════════════

_CV2_COLORMAP_TABLE = {
    "autumn": cv2.COLORMAP_AUTUMN, "bone": cv2.COLORMAP_BONE,
    "jet": cv2.COLORMAP_JET, "winter": cv2.COLORMAP_WINTER,
    "rainbow": cv2.COLORMAP_RAINBOW, "ocean": cv2.COLORMAP_OCEAN,
    "summer": cv2.COLORMAP_SUMMER, "spring": cv2.COLORMAP_SPRING,
    "cool": cv2.COLORMAP_COOL, "hot": cv2.COLORMAP_HOT,
    "pink": cv2.COLORMAP_PINK, "hsv": cv2.COLORMAP_HSV,
    "parula": cv2.COLORMAP_PARULA, "magma": cv2.COLORMAP_MAGMA,
    "inferno": cv2.COLORMAP_INFERNO, "plasma": cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS, "cividis": cv2.COLORMAP_CIVIDIS,
    "turbo": cv2.COLORMAP_TURBO, "twilight": cv2.COLORMAP_TWILIGHT,
    "deepgreen": cv2.COLORMAP_DEEPGREEN,
}


def resolve_cv2_colormap(name: str) -> int:
    """Resolve a colormap name to an OpenCV constant."""
    key = name.lower().strip()
    if key in _CV2_COLORMAP_TABLE:
        return _CV2_COLORMAP_TABLE[key]
    print(f"[WARN] Unknown colormap '{name}', falling back to INFERNO. "
          f"Available: {sorted(_CV2_COLORMAP_TABLE.keys())}")
    return cv2.COLORMAP_INFERNO


def heightmap_to_bgr(
    hmap_m: np.ndarray,
    vmin: float,
    vmax: float,
    colormap_id: int,
    invert: bool = False,
) -> np.ndarray:
    """Convert a (H, W) heightmap in meters to (H, W, 3) uint8 BGR.

    Args:
        hmap_m:  float32 heightmap (meters), shape (H, W).
        vmin:    clamp floor (meters).
        vmax:    clamp ceiling (meters).
        colormap_id: cv2.COLORMAP_* constant.
        invert:  if True, high heights map to dark colours.

    Returns:
        BGR uint8 image (H, W, 3).
    """
    d = np.clip(hmap_m.astype(np.float64), vmin, vmax)
    norm = ((d - vmin) / max(vmax - vmin, 1e-8) * 255.0).astype(np.uint8)
    if invert:
        norm = 255 - norm
    return cv2.applyColorMap(norm, colormap_id)


# ═══════════════════════════════════════════════════════════════════════
# USD helpers
# ═══════════════════════════════════════════════════════════════════════

def debug_box_sync_prims(stage, prim_path: str):
    """Inspect and fix visibility/purpose on a box prim and its children.

    Guarded by --debug_box_sync at call sites.  If an Imageable prim
    has purpose != 'default' or visibility == 'invisible', it is force-
    fixed so the box is guaranteed to render.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"  [BOX_SYNC_DBG] prim {prim_path} is INVALID")
        return

    for child_prim in [prim] + list(prim.GetAllChildren()):
        p = child_prim.GetPath().pathString
        img = UsdGeom.Imageable(child_prim)
        if not img:
            continue
        purpose_attr = img.GetPurposeAttr()
        purpose = purpose_attr.Get() if purpose_attr else None
        vis_attr = img.GetVisibilityAttr()
        vis = vis_attr.Get() if vis_attr else None
        print(f"  [BOX_SYNC_DBG]   {p}  purpose={purpose}  visibility={vis}")

        needs_fix = False
        if purpose is not None and purpose != UsdGeom.Tokens.default_:
            print(f"  [BOX_SYNC_DBG]     → fixing purpose to 'default'")
            purpose_attr.Set(UsdGeom.Tokens.default_)
            needs_fix = True
        if vis is not None and vis == UsdGeom.Tokens.invisible:
            print(f"  [BOX_SYNC_DBG]     → calling MakeVisible()")
            img.MakeVisible()
            needs_fix = True
        if needs_fix:
            print(f"  [BOX_SYNC_DBG]     FIXED: {p}")


def set_kinematic(stage, prim_path: str, kinematic: bool):
    """Toggle kinematic_enabled on a rigid body via UsdPhysics.RigidBodyAPI.

    Robust: applies RigidBodyAPI if missing, uses Create (not Get) to ensure
    the attribute always exists.  Prevents silent no-ops that can cause
    invisible boxes or CCD-on-kinematic warnings.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)
    rb = UsdPhysics.RigidBodyAPI.Get(stage, prim_path)
    rb.CreateKinematicEnabledAttr().Set(bool(kinematic))


def set_disable_gravity(stage, prim_path: str, disable: bool):
    """Toggle disable_gravity on a rigid body via PhysxSchema."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    attr = api.GetDisableGravityAttr()
    if not attr or not attr.IsValid():
        attr = api.CreateDisableGravityAttr()
    attr.Set(bool(disable))


def tune_rigid_body(stage, prim_path: str, *,
                    lin_damp=2.0, ang_damp=2.0,
                    max_depen_vel=0.5, max_lin_vel=2.0,
                    pos_iters=16, vel_iters=4):
    """Set PhysX rigid body tuning parameters for stable contacts."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    for getter, creator, val in [
        ("GetLinearDampingAttr",  "CreateLinearDampingAttr",  float(lin_damp)),
        ("GetAngularDampingAttr", "CreateAngularDampingAttr", float(ang_damp)),
        ("GetMaxDepenetrationVelocityAttr", "CreateMaxDepenetrationVelocityAttr", float(max_depen_vel)),
        ("GetMaxLinearVelocityAttr", "CreateMaxLinearVelocityAttr", float(max_lin_vel)),
        ("GetSolverPositionIterationCountAttr", "CreateSolverPositionIterationCountAttr", int(pos_iters)),
        ("GetSolverVelocityIterationCountAttr", "CreateSolverVelocityIterationCountAttr", int(vel_iters)),
    ]:
        attr = getattr(api, getter)()
        if not attr or not attr.IsValid():
            attr = getattr(api, creator)()
        attr.Set(val)


def set_physics_material(stage, prim_path: str, *,
                         static_friction=1.2, dynamic_friction=0.9,
                         restitution=0.0,
                         restitution_combine_mode="min"):
    """Apply/update a UsdPhysics material on a prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    mat_api = UsdPhysics.MaterialAPI.Apply(prim)
    mat_api.CreateStaticFrictionAttr().Set(float(static_friction))
    mat_api.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    mat_api.CreateRestitutionAttr().Set(float(restitution))
    # PhysX restitution combine mode: "average", "min", "multiply", "max"
    api = PhysxSchema.PhysxMaterialAPI.Apply(prim)
    attr = api.GetRestitutionCombineModeAttr()
    if not attr or not attr.IsValid():
        attr = api.CreateRestitutionCombineModeAttr()
    attr.Set(restitution_combine_mode)


# ═══════════════════════════════════════════════════════════════════════
# AABB overlap check
# ═══════════════════════════════════════════════════════════════════════

def aabb_overlap(pos_a, dims_a, pos_b, dims_b, margin: float = 0.005) -> bool:
    """Check if two axis-aligned bounding boxes overlap (with a small margin).

    Boxes are assumed to be axis-aligned (yaw 0 or 90° → already rotated into
    dims).  This is intentionally conservative — a small margin prevents
    false-positive triggers from touching-but-not-penetrating surfaces.

    Returns True if boxes overlap beyond margin.
    """
    for axis in range(3):
        half_a = 0.5 * dims_a[axis] - margin
        half_b = 0.5 * dims_b[axis] - margin
        if abs(pos_a[axis] - pos_b[axis]) >= half_a + half_b:
            return False
    return True

def aabb_intersection_area(pos_a, dims_a, pos_b, dims_b) -> float:
    """Compute the 2D intersection area (XY) of two AABBs."""
    x_overlap = max(0.0, min(pos_a[0] + dims_a[0]/2, pos_b[0] + dims_b[0]/2) - 
                         max(pos_a[0] - dims_a[0]/2, pos_b[0] - dims_b[0]/2))
    y_overlap = max(0.0, min(pos_a[1] + dims_a[1]/2, pos_b[1] + dims_b[1]/2) - 
                         max(pos_a[1] - dims_a[1]/2, pos_b[1] - dims_b[1]/2))
    return float(x_overlap * y_overlap)


# ═══════════════════════════════════════════════════════════════════════
# Deterministic grid packing
# ═══════════════════════════════════════════════════════════════════════

def generate_placements(num_boxes: int, pallet_lx: float, pallet_ly: float,
                        pallet_top_z: float, seed: int = 42):
    """
    Generate plausible box placements using greedy grid packing.

    Returns list of dicts with keys: dims (after yaw), target_xyz, yaw,
    color_rgb.
    """
    rng = np.random.default_rng(seed)
    placements: list[dict] = []

    # Split into 2 layers
    layer0_count = min(num_boxes, num_boxes // 2 + num_boxes % 2)
    layer1_count = num_boxes - layer0_count

    # Track placed AABBs per layer for overlap avoidance
    layer0_tops: list[tuple[np.ndarray, np.ndarray]] = []  # (pos3, dims3)

    half_lx = 0.5 * pallet_lx
    half_ly = 0.5 * pallet_ly

    for layer_idx in range(2):
        count = layer0_count if layer_idx == 0 else layer1_count
        if count == 0:
            continue

        # Layer base z
        if layer_idx == 0:
            layer_base_z = pallet_top_z
        else:
            if layer0_tops:
                layer_base_z = max(p[2] + 0.5 * d[2] for p, d in layer0_tops)
            else:
                layer_base_z = pallet_top_z + 0.20

        for bi in range(count):
            # Random box dims (realistic cardboard range)
            bx = float(rng.uniform(0.22, 0.38))
            by = float(rng.uniform(0.18, 0.30))
            bz = float(rng.uniform(0.12, 0.22))

            # Random yaw (0 or 90°)
            yaw = float(rng.choice([0.0, math.pi / 2.0]))

            # Effective footprint (after yaw)
            if abs(yaw) > 0.1:
                eff_x, eff_y = by, bx
            else:
                eff_x, eff_y = bx, by
            dims = np.array([eff_x, eff_y, bz], dtype=np.float32)

            # Greedy placement: try up to 50 random positions, pick first
            # non-overlapping one
            margin_x = 0.5 * eff_x + 0.01
            margin_y = 0.5 * eff_y + 0.01
            tz = layer_base_z + 0.5 * bz

            best_pos = None
            for _ in range(50):
                tx = float(rng.uniform(-half_lx + margin_x, half_lx - margin_x))
                ty = float(rng.uniform(-half_ly + margin_y, half_ly - margin_y))
                candidate = np.array([tx, ty, tz], dtype=np.float32)

                # 1) Check overlap with same-layer boxes
                overlap = False
                for prev_pos, prev_dims in (layer0_tops if layer_idx == 0 else
                                             [(p["target_xyz"], p["dims"]) for p in placements[layer0_count:]]):
                    if aabb_overlap(candidate, dims, prev_pos, prev_dims, margin=0.01):
                        overlap = True
                        break
                if overlap:
                    continue

                # 2) For layer > 0, check support from layer below
                if layer_idx > 0:
                    box_area = float(dims[0] * dims[1])
                    supported_area = 0.0
                    for prev_pos, prev_dims in layer0_tops:
                        supported_area += aabb_intersection_area(
                            candidate, dims, prev_pos, prev_dims
                        )
                    if supported_area / box_area < args.support_ratio_min:
                        continue  # Not enough support

                # If we made it here, placement is valid!
                best_pos = candidate
                break

            if best_pos is None:
                # Skip this box if we can't find a valid supported pos in 50 tries
                continue

            placements.append({
                "dims": dims,
                "target_xyz": best_pos,
                "yaw": yaw,
                "color_rgb": rng.uniform(0.3, 0.9, size=(3,)).astype(np.float32),
            })

            if layer_idx == 0:
                layer0_tops.append((best_pos.copy(), dims.copy()))

    return placements


# ═══════════════════════════════════════════════════════════════════════
# Box colors (for visual variety)
# ═══════════════════════════════════════════════════════════════════════

BOX_COLORS = [
    (0.85, 0.65, 0.40),  # Cardboard
    (0.75, 0.55, 0.35),  # Kraft
    (0.90, 0.80, 0.60),  # Sandy
    (0.70, 0.50, 0.30),  # Walnut
    (0.65, 0.55, 0.40),  # Medium brown
    (0.80, 0.70, 0.50),  # Tan
    (0.55, 0.45, 0.35),  # Dark brown
    (0.30, 0.50, 0.70),  # Blue (marked)
    (0.70, 0.30, 0.30),  # Red (fragile)
    (0.40, 0.60, 0.40),  # Green
]


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    import omni.usd

    device = args.device

    placement_mode = args.placement_mode

    print(f"\n{'='*60}")
    print(f"  Palletiser Mockup Video — {placement_mode.upper()} Mode")
    print(f"{'='*60}")
    print(f"  Output:       {args.output_path}")
    print(f"  FPS:          {args.fps}")
    print(f"  Duration:     {args.duration_s}s")
    print(f"  Boxes:        {args.num_boxes}")
    print(f"  Seed:         {args.seed}")
    print(f"  Placement:    {placement_mode}")
    print(f"  Substeps:     {args.sim_substeps}")
    if placement_mode == "drop":
        print(f"  Lower speed:  {args.lower_speed} m/s")
        print(f"  Release gap:  {args.release_clearance} m")
        print(f"  Settle time:  {args.settle_s} s")
        print(f"  Freeze:       {args.freeze_after_settle}")
    print(f"  Debug:        {args.debug}")
    print(f"  Record mode:  {args.record_mode}")
    if args.record_mode in ("heightmap", "both"):
        _v = args.hmap_vmax if args.hmap_vmax > 0 else "auto"
        print(f"  Hmap range:   [{args.hmap_vmin}, {_v}] m")
        print(f"  Hmap cmap:    {args.hmap_colormap}")
        print(f"  Noise off:    {not args.enable_depth_noise}")
    print(f"{'='*60}\n")

    # ─── Environment config ───────────────────────────────────────────
    cfg = PalletTaskCfg()
    cfg.scene.num_envs = 1
    cfg.sim.render_interval = 1
    
    if args.physics_device == "cpu":
        cfg.sim.device = "cpu"
        print("[INFO] Running physics on CPU (--physics_device=cpu)")
    else:
        cfg.sim.device = device
        print(f"[INFO] Running physics on GPU ({device})")
        
    # GPU buffers do not grow dynamically and can crash under heavy contact.
    # Increasing capacities manually prevents PhysX from overflowing its buffers.
    cfg.sim.physx.gpu_found_lost_pairs_capacity = args.gpu_found_lost_pairs_capacity
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = args.gpu_total_aggregate_pairs_capacity
    cfg.sim.physx.gpu_heap_capacity = args.gpu_heap_capacity
    cfg.sim.physx.gpu_temp_buffer_capacity = args.gpu_temp_buffer_capacity
    cfg.max_boxes = max(args.num_boxes + 10, 50)  # extra for retries
    cfg.decimation = 1

    # Enable visual features
    cfg.use_pallet_mesh_visual = True
    cfg.floor_visual_enabled = True
    cfg.mockup_mode = True

    # Camera
    cfg.scene.render_camera.width = args.cam_width
    cfg.scene.render_camera.height = args.cam_height

    # Pallet geometry constants (from PalletSceneCfg)
    pallet_center_z = cfg.scene.pallet.init_state.pos[2]   # 0.075
    pallet_half_h = cfg.scene.pallet.spawn.size[2] / 2.0   # 0.075
    PALLET_TOP_Z = pallet_center_z + pallet_half_h          # 0.15
    PALLET_LX = cfg.scene.pallet.spawn.size[0]              # 1.2
    PALLET_LY = cfg.scene.pallet.spawn.size[1]              # 0.8

    print(f"[INFO] Pallet: {PALLET_LX}x{PALLET_LY}, top_z={PALLET_TOP_Z:.3f}")

    # ─── Create environment ───────────────────────────────────────────
    env = PalletTask(cfg, render_mode="rgb_array")

    # FIX: reset env so that PhysX views, scene buffers, and sensors are
    # fully initialised before we attempt to write box poses.
    print("[INFO] Calling env.reset() to initialise PhysX views ...")
    env.reset()

    # Warmup: step + update a few times so that PhysX GPU broadphase,
    # render products, and depth-camera buffers are all fully active.
    _warmup_dt = env.sim.get_physics_dt()
    for _wi in range(5):
        env.sim.step()
        env.scene.update(dt=_warmup_dt)
        # CRITICAL: pump the Kit renderer so render products are initialized
        simulation_app.update()
    print("[INFO] Warmup complete (5 sim steps + render pumps).")

    stage = omni.usd.get_context().get_stage()

    # ─── Depth camera access (robust, with fallbacks) ─────────────────
    needs_depth = args.record_mode in ("heightmap", "both")
    depth_cam = None
    hmap_converter = None
    if needs_depth:
        # Try 1: env.scene["depth_camera"]
        try:
            depth_cam = env.scene["depth_camera"]
            print("[INFO] Depth camera found via env.scene['depth_camera']")
        except (KeyError, AttributeError):
            pass
        # Try 2: env.scene.sensors["depth_camera"]
        if depth_cam is None:
            try:
                depth_cam = env.scene.sensors["depth_camera"]
                print("[INFO] Depth camera found via env.scene.sensors['depth_camera']")
            except (KeyError, AttributeError):
                pass
        # Fail with diagnostics
        if depth_cam is None:
            _keys = []
            try:
                _keys.extend(list(env.scene.keys()))
            except Exception:
                pass
            try:
                _keys.extend([f"sensors.{k}" for k in env.scene.sensors.keys()])
            except Exception:
                pass
            raise RuntimeError(
                f"'depth_camera' was not found in the scene. "
                f"Available keys: {_keys}"
            )
        print(f"[INFO] Depth camera type: {type(depth_cam).__name__}")

        # ─── DepthHeightmapConverter (recording-only, noise off by default) ─
        # Noise is disabled unless user explicitly passes --enable_depth_noise
        _noise_on = args.enable_depth_noise
        depth_hmap_cfg = DepthHeightmapCfg(
            cam_height=cfg.depth_cam_resolution[0],
            cam_width=cfg.depth_cam_resolution[1],
            fov_deg=cfg.depth_cam_fov_deg,
            sensor_height_m=cfg.depth_cam_height_m,
            map_h=cfg.map_shape[0],
            map_w=cfg.map_shape[1],
            crop_x=cfg.depth_crop_x,
            crop_y=cfg.depth_crop_y,
            noise_enable=_noise_on,
            noise_sigma_m=cfg.depth_noise_sigma_m,
            noise_scale=cfg.depth_noise_scale,
            noise_quantization_m=cfg.depth_noise_quantization_m,
            noise_dropout_prob=cfg.depth_noise_dropout_prob,
        )
        hmap_converter = DepthHeightmapConverter(depth_hmap_cfg, device=device)
        print(f"[INFO] Heightmap converter created (noise_enable={_noise_on})")

        # ── Set depth camera look-at: top-down over pallet center ──
        try:
            _dc_device = torch.device(device)
            _dc_eyes = torch.tensor([[0.0, 0.0, cfg.depth_cam_height_m]],
                                    device=_dc_device).repeat(cfg.scene.num_envs, 1)
            _dc_targets = torch.tensor([[0.0, 0.0, 0.0]],
                                       device=_dc_device).repeat(cfg.scene.num_envs, 1)
            depth_cam.set_world_poses_from_view(eyes=_dc_eyes, targets=_dc_targets)
            print(f"[INFO] Depth camera look-at set: eye=(0,0,{cfg.depth_cam_height_m}), "
                  f"target=(0,0,0) [top-down]")
        except Exception as _e:
            print(f"[WARN] Could not set depth camera look-at: {_e}")

    boxes = env.scene["boxes"]

    # ─── GPU alignment check ───────────────────────────────────────────
    try:
        import carb.settings
        _settings = carb.settings.get_settings()
        _render_gpu = _settings.get("/renderer/activeGpu")
        _physics_gpu = _settings.get("/physics/cudaDevice")
        print(f"[INFO] renderer/activeGpu  = {_render_gpu}  (Vulkan ordinal)")
        print(f"[INFO] physics/cudaDevice  = {_physics_gpu}  (CUDA ordinal)")
        # NOTE: These use different ordinal namespaces and are expected to
        #       differ when Vulkan and CUDA enumerate GPUs in different order.
        if _physics_gpu != 0:
            print("[WARN] physics/cudaDevice is not 0 — RTX 6000 should be "
                  "CUDA device 0. PhysX may target an unsupported GPU!")
    except Exception as _e:
        print(f"[WARN] Could not read GPU alignment settings: {_e}")

    # ─── Tune all box prims for stable, non-elastic contacts ──────────
    #     Also: disable CCD (causes GPU crash with kinematic toggling)
    #           and apply bright debug material for visibility.
    _debug_mat_path = "/World/_DebugBoxMaterial"
    _debug_mat_prim = stage.GetPrimAtPath(_debug_mat_path)
    if not _debug_mat_prim.IsValid():
        mat = UsdShade.Material.Define(stage, _debug_mat_path)
        shader = UsdShade.Shader.Define(stage, f"{_debug_mat_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.85, 0.25, 0.20)   # bright red — unmissable
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    _debug_mat = UsdShade.Material.Get(stage, _debug_mat_path)

    for i in range(cfg.max_boxes):
        bp = f"/World/envs/env_0/Boxes/box_{i}"
        box_prim = stage.GetPrimAtPath(bp)
        if not box_prim.IsValid():
            continue
        tune_rigid_body(stage, bp,
                        lin_damp=2.0, ang_damp=3.0,
                        max_depen_vel=0.3, max_lin_vel=1.5,
                        pos_iters=32, vel_iters=8)
        set_physics_material(stage, bp,
                             static_friction=1.2,
                             dynamic_friction=0.9,
                             restitution=0.0,
                             restitution_combine_mode="min")
        # ── Patch B: disable CCD to prevent GPU narrowphase crash ──
        _rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(box_prim)
        _rb_api.CreateEnableCCDAttr().Set(False)
        # ── Patch E: bind bright debug material for visibility ──
        UsdShade.MaterialBindingAPI.Apply(box_prim).Bind(_debug_mat)
        # ── Patch V: displayColor/Opacity fallback (Storm ignores unresolved materials) ──
        _gprim = UsdGeom.Gprim(box_prim)
        if _gprim:
            _gprim.CreateDisplayColorAttr().Set(
                [Gf.Vec3f(0.85, 0.25, 0.20)]   # bright red
            )
            _gprim.CreateDisplayOpacityAttr().Set([1.0])

    print("[INFO] CCD disabled for all box prims (stability mode)")
    print(f"[INFO] Applied debug material + displayColor to {cfg.max_boxes} boxes")

    # ─── Debug diagnostics ─────────────────────────────────────────────
    if args.debug:
        print(f"\n[DEBUG] === Prim Diagnostics ===")
        _max_debug = min(5, cfg.max_boxes)
        for _di in range(_max_debug):
            _dp = f"/World/envs/env_0/Boxes/box_{_di}"
            _dprim = stage.GetPrimAtPath(_dp)
            _valid = _dprim.IsValid() if _dprim else False
            print(f"  prim[{_di}] = {_dp}  valid={_valid}")
            if _valid:
                _img = UsdGeom.Imageable(_dprim)
                _vis = _img.GetVisibilityAttr().Get() if _img else "N/A"
                print(f"           visibility = {_vis}")
        print(f"[DEBUG] === End Prim Diagnostics ===\n")

    # ─── Plan placements ──────────────────────────────────────────────
    placements = generate_placements(
        args.num_boxes, PALLET_LX, PALLET_LY, PALLET_TOP_Z, seed=args.seed
    )
    print(f"[INFO] Planned {len(placements)} placements")

    # ─── Helpers: pose & velocity ───────────────────────────────────
    # Cached env_ids tensor for single-env mockup (env 0 only)
    _env_ids_0 = torch.tensor([0], dtype=torch.long, device=device)

    def set_box_pose(idx: int, pos_xyz, yaw_rad: float):
        """Write full state for box `idx` in env 0 directly to PhysX.

        Uses write_object_state_to_sim() which is the only API that
        reliably pushes transforms into the simulation.  write_data_to_sim()
        only flushes external wrenches and does NOT update poses.
        """
        px, py, pz = float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])
        cy = math.cos(0.5 * yaw_rad)
        sy = math.sin(0.5 * yaw_rad)

        # State: [pos(3), quat_wxyz(4), lin_vel(3), ang_vel(3)] = 13
        state = torch.zeros(1, 1, 13, dtype=torch.float32, device=device)
        state[0, 0, 0] = px
        state[0, 0, 1] = py
        state[0, 0, 2] = pz
        state[0, 0, 3] = cy   # qw
        state[0, 0, 4] = 0.0  # qx
        state[0, 0, 5] = 0.0  # qy
        state[0, 0, 6] = sy   # qz
        # lin_vel and ang_vel are already zero

        object_ids = torch.tensor([idx], dtype=torch.long, device=device)
        boxes.write_object_state_to_sim(state, _env_ids_0, object_ids)

        # --debug_box_sync: readback verification after every write
        if args.debug_box_sync:
            env.sim.step()
            env.scene.update(dt=env.sim.get_physics_dt())
            rb_pos = boxes.data.object_pos_w[0, idx].cpu()
            all_z = boxes.data.object_pos_w[0, :args.num_boxes, 2].cpu()
            print(f"  [BOX_SYNC] set_box_pose({idx}) "
                  f"target=({px:.3f},{py:.3f},{pz:.3f})  "
                  f"readback=({rb_pos[0]:.3f},{rb_pos[1]:.3f},{rb_pos[2]:.3f})  "
                  f"all_z: min={all_z.min():.2f} max={all_z.max():.2f}")

    def zero_box_vel(idx: int):
        """Zero velocities for box `idx` by re-writing its full state.

        Reads back the current pose from boxes.data.* (which the sim
        updated on the last scene.update()) and writes the same pose
        with zero velocities.
        """
        cur_pos = boxes.data.object_pos_w[0, idx]    # (3,)
        cur_quat = boxes.data.object_quat_w[0, idx]  # (4,) wxyz

        state = torch.zeros(1, 1, 13, dtype=torch.float32, device=device)
        state[0, 0, :3] = cur_pos
        state[0, 0, 3:7] = cur_quat
        # vel fields are already zero

        object_ids = torch.tensor([idx], dtype=torch.long, device=device)
        boxes.write_object_state_to_sim(state, _env_ids_0, object_ids)

    def get_box_pos(idx: int) -> np.ndarray:
        """Read current position of box `idx`."""
        return boxes.data.object_pos_w[0, idx].cpu().numpy()

    def get_box_vel_norm(idx: int) -> float:
        """Combined linear + angular velocity magnitude."""
        lv = boxes.data.object_lin_vel_w[0, idx].cpu().numpy()
        av = boxes.data.object_ang_vel_w[0, idx].cpu().numpy()
        return float(np.linalg.norm(lv) + 0.2 * np.linalg.norm(av))

    def box_prim_path(idx: int) -> str:
        return f"/World/envs/env_0/Boxes/box_{idx}"

    # ─── Sim stepping + capture ───────────────────────────────────────
    sim_dt = env.sim.get_physics_dt()
    substeps = args.sim_substeps
    frame_dt = 1.0 / args.fps

    frames: list[np.ndarray] = []
    depth_frame_idx = 0  # counter for raw depth dump filenames

    # Optional raw depth output directory
    raw_dir: str | None = None
    if needs_depth and args.save_depth_raw:
        if args.depth_raw_dir != "":
            raw_dir = args.depth_raw_dir
        else:
            raw_dir = os.path.splitext(args.output_path)[0] + "_depth_raw"
        os.makedirs(raw_dir, exist_ok=True)
        print(f"[INFO] Raw depth frames will be saved to: {raw_dir}")

    # Resolve heightmap visualization parameters
    hmap_vmin = args.hmap_vmin
    hmap_vmax = args.hmap_vmax if args.hmap_vmax > 0 else cfg.max_height
    hmap_cmap_id = resolve_cv2_colormap(args.hmap_colormap)
    hmap_invert = args.hmap_invert

    def _capture_heightmap_bgr() -> np.ndarray | None:
        """Read depth camera, convert to heightmap, return BGR frame."""
        nonlocal depth_frame_idx

        # CRITICAL: explicitly update depth camera so it reads fresh data
        depth_cam.update(dt=sim_dt * substeps)

        depth_raw = depth_cam.data.output["distance_to_image_plane"]
        # To torch tensor (N, H, W)
        if hasattr(depth_raw, 'detach'):
            depth_t = depth_raw.detach()
        else:
            depth_t = torch.as_tensor(depth_raw, device=device)
        if depth_t.dim() == 4 and depth_t.shape[-1] == 1:
            depth_t = depth_t.squeeze(-1)
        if depth_t.dim() == 2:
            depth_t = depth_t.unsqueeze(0)  # add batch dim

        # ── Depth diagnostics (every 50 frames) ──
        if depth_frame_idx % 50 == 0:
            _d0 = depth_t[0].cpu().numpy()
            _valid_mask = _d0 > 0.01
            _valid_count = int(_valid_mask.sum())
            _total = _d0.size
            if _valid_count > 0:
                _d_min = float(_d0[_valid_mask].min())
                _d_max = float(_d0[_valid_mask].max())
                _d_mean = float(_d0[_valid_mask].mean())
            else:
                _d_min = _d_max = _d_mean = 0.0
            print(f"  [DEPTH DBG] frame={depth_frame_idx} "
                  f"valid={_valid_count}/{_total} "
                  f"min={_d_min:.3f} max={_d_max:.3f} mean={_d_mean:.3f}")

        # Camera pose
        cam_pos = depth_cam.data.pos_w       # (N, 3)
        cam_quat = depth_cam.data.quat_w_world  # (N, 4) wxyz

        # Depth → heightmap via converter
        hmap_t = hmap_converter.depth_to_heightmap(depth_t, cam_pos, cam_quat)
        hmap_np = hmap_t[0].cpu().numpy()  # (H, W) meters

        # Optional raw depth dump
        if raw_dir is not None:
            depth_np = depth_t[0].cpu().numpy()
            np.save(
                os.path.join(raw_dir, f"depth_{depth_frame_idx:06d}.npy"),
                depth_np.astype(np.float32),
            )
        depth_frame_idx += 1

        # Heightmap → colormapped BGR
        bgr = heightmap_to_bgr(hmap_np, hmap_vmin, hmap_vmax,
                               hmap_cmap_id, hmap_invert)
        return bgr

    def step_and_capture():
        """Run physics substeps, update scene, render one frame."""
        for _ in range(substeps):
            env.sim.step()
        env.scene.update(dt=sim_dt * substeps)

        # CRITICAL: Flush USD transforms into the renderer.  In headless mode,
        # without this call camera render products never see the updated
        # prim transforms — resulting in invisible/stale geometry.
        simulation_app.update()

        record_mode = args.record_mode

        if record_mode == "rgb":
            # ── RGB only (original behaviour) ──
            fr = env.render()
            if fr is not None:
                frames.append(fr)

        elif record_mode == "heightmap":
            # ── Heightmap only ──
            bgr = _capture_heightmap_bgr()
            if bgr is not None:
                frames.append(bgr)

        elif record_mode == "both":
            # ── Side-by-side: RGB (left) + heightmap (right) ──
            rgb_frame = env.render()
            hmap_bgr = _capture_heightmap_bgr()
            if rgb_frame is not None and hmap_bgr is not None:
                # Convert RGB to BGR for consistency
                rgb_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                rgb_h, rgb_w = rgb_bgr.shape[:2]
                hm_h, hm_w = hmap_bgr.shape[:2]
                # Resize heightmap to match RGB height, preserving aspect ratio
                if hm_h != rgb_h:
                    scale = rgb_h / hm_h
                    new_w = int(hm_w * scale)
                    hmap_bgr = cv2.resize(hmap_bgr, (new_w, rgb_h),
                                          interpolation=cv2.INTER_LINEAR)
                composite = np.concatenate([rgb_bgr, hmap_bgr], axis=1)
                frames.append(composite)
            elif rgb_frame is not None:
                # FIX: heightmap capture failed — use a black placeholder
                # of the expected heightmap size so every frame has
                # identical dimensions (prevents VideoWriter crash).
                rgb_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                rgb_h, rgb_w = rgb_bgr.shape[:2]
                # Compute expected heightmap width after scaling to RGB height
                raw_hm_h, raw_hm_w = cfg.map_shape  # (H, W)
                scale = rgb_h / raw_hm_h
                placeholder_w = int(raw_hm_w * scale)
                placeholder = np.zeros((rgb_h, placeholder_w, 3), dtype=np.uint8)
                composite = np.concatenate([rgb_bgr, placeholder], axis=1)
                frames.append(composite)
    # ─── Park all boxes out of view at start ──────────────────────────
    park_pos = [0.0, 0.0, -5.0]
    for i in range(cfg.max_boxes):
        set_box_pose(i, park_pos, 0.0)
        bp = box_prim_path(i)
        set_kinematic(stage, bp, True)
        set_disable_gravity(stage, bp, True)

    # Let the scene settle for a few frames
    for _ in range(10):
        step_and_capture()

    # ─── Diagnostic frame dump (always, not gated) ─────────────────────
    # Saves exactly 1 RGB + 1 heightmap PNG after the first placement
    # to the output directory for quick visual verification.
    _diag_dir = os.path.dirname(args.output_path) or "."
    os.makedirs(_diag_dir, exist_ok=True)
    _diag_saved = False  # only dump once

    # ─── State machine ────────────────────────────────────────────────
    total_frames = int(args.duration_s * args.fps)
    frame_count = 0

    placed_boxes: list[tuple[np.ndarray, np.ndarray]] = []  # (pos, dims)
    current_box_idx = 0    # which box prim to use
    placement_idx = 0      # which placement plan we are on
    retry_count = 0

    # State machine state
    state = "SPAWN"
    state_timer = 0.0
    carry_start_pos = np.array([1.0, -0.8, args.carry_height], dtype=np.float32)

    # Interpolation frame counter for PLACE_KINEMATIC
    place_kin_frame = 0
    PLACE_KIN_FRAMES = 15  # number of frames for smooth kinematic placement

    print(f"\n[INFO] Starting animation: {total_frames} frames "
          f"(mode={placement_mode})...")

    while frame_count < total_frames:
        # ── Only process if we have placements left ──
        if placement_idx < len(placements) and current_box_idx < cfg.max_boxes:
            pl = placements[placement_idx]
            dims = pl["dims"]
            target = pl["target_xyz"]
            yaw = pl["yaw"]
            bp = box_prim_path(current_box_idx)

            # ─────────────────────────────────────────
            if state == "SPAWN":
                # Teleport box to side (far from any geometry), kinematic
                carry_start_pos = np.array(
                    [PALLET_LX + 0.5, -0.5, args.carry_height],
                    dtype=np.float32,
                )
                set_kinematic(stage, bp, True)
                set_disable_gravity(stage, bp, True)
                set_box_pose(current_box_idx, carry_start_pos.tolist(), yaw)

                if args.debug:
                    print(f"  [DEBUG] SPAWN box_idx={current_box_idx} "
                          f"placement={placement_idx+1}/{len(placements)} "
                          f"prim={bp}")

                state = "CARRY"
                state_timer = 0.0

            # ─────────────────────────────────────────
            elif state == "CARRY":
                # Smooth ease-in-out interpolation to above target
                state_timer += frame_dt
                carry_dur = 0.8  # seconds
                alpha = min(1.0, state_timer / carry_dur)
                # Smooth step (ease-in-out)
                alpha_smooth = 0.5 - 0.5 * math.cos(math.pi * alpha)

                above = np.array(
                    [target[0], target[1], args.carry_height],
                    dtype=np.float32,
                )
                pos = (1.0 - alpha_smooth) * carry_start_pos + alpha_smooth * above
                set_box_pose(current_box_idx, pos.tolist(), yaw)

                if alpha >= 1.0:
                    state = "LOWER"
                    state_timer = 0.0

            # ─────────────────────────────────────────
            elif state == "LOWER":
                # Kinematic descent at controlled speed, still kinematic
                cur_pos = get_box_pos(current_box_idx)

                if placement_mode == "kinematic":
                    # In kinematic mode, lower directly to target Z
                    final_z = float(target[2])
                else:
                    # In drop mode, stop slightly above target for release
                    final_z = float(target[2]) + args.release_clearance

                new_z = max(final_z,
                            cur_pos[2] - args.lower_speed * frame_dt)
                pos = [float(target[0]), float(target[1]), new_z]
                set_box_pose(current_box_idx, pos, yaw)

                if abs(new_z - final_z) < 1e-4:
                    if placement_mode == "kinematic":
                        # ── Kinematic: place directly, no physics ──
                        if args.debug:
                            _pre = get_box_pos(current_box_idx)
                            print(f"  [DEBUG] Pre-place pos: "
                                  f"({_pre[0]:.3f}, {_pre[1]:.3f}, {_pre[2]:.3f})")
                        set_box_pose(current_box_idx,
                                     [float(target[0]), float(target[1]),
                                      float(target[2])], yaw)
                        place_kin_frame = 0
                        state = "PLACE_KINEMATIC"
                        state_timer = 0.0
                    else:
                        # ── Drop: release to dynamic physics ──
                        set_kinematic(stage, bp, False)
                        set_disable_gravity(stage, bp, False)
                        zero_box_vel(current_box_idx)
                        state = "SETTLE"
                        state_timer = 0.0

            # ─────────────────────────────────────────
            # KINEMATIC-ONLY: smooth arrival at target + immediate freeze
            elif state == "PLACE_KINEMATIC":
                place_kin_frame += 1
                # Box is already at target pose & kinematic; just wait
                # a few frames so the placement looks intentional.
                if place_kin_frame >= PLACE_KIN_FRAMES:
                    final_pos = get_box_pos(current_box_idx)
                    # Keep kinematic, gravity off — box stays in place
                    zero_box_vel(current_box_idx)

                    placed_boxes.append((final_pos.copy(), dims.copy()))
                    placement_idx += 1
                    current_box_idx += 1
                    retry_count = 0
                    state = "PAUSE"
                    state_timer = 0.0
                    print(f"  ✓ Placed box {placement_idx}/{len(placements)} "
                          f"at ({final_pos[0]:.2f}, {final_pos[1]:.2f}, "
                          f"{final_pos[2]:.2f})  [kinematic]")
                    # --debug_box_sync: verify the placed box reached sim
                    if args.debug_box_sync:
                        # The placed box is current_box_idx - 1
                        # (current_box_idx was already incremented above)
                        _placed_idx = current_box_idx - 1
                        env.sim.step()
                        env.scene.update(dt=sim_dt)
                        rb = boxes.data.object_pos_w[0, _placed_idx].cpu()
                        target_z = final_pos[2]
                        tag = "OK" if rb[2] > -1.0 else "STUCK at parking z!"
                        print(f"  [BOX_SYNC] box {_placed_idx} "
                              f"target_z={target_z:.4f}  "
                              f"readback=({rb[0]:.3f},{rb[1]:.3f},{rb[2]:.3f})  "
                              f"({tag})")
                        debug_box_sync_prims(stage, box_prim_path(_placed_idx))

                    if args.debug:
                        # Print visibility of first placed box
                        if placement_idx == 1:
                            _check_prim = stage.GetPrimAtPath(
                                box_prim_path(current_box_idx - 1))
                            if _check_prim.IsValid():
                                _img = UsdGeom.Imageable(_check_prim)
                                _vis = _img.GetVisibilityAttr().Get()
                                print(f"  [DEBUG] First box visibility = {_vis}")

            # ─────────────────────────────────────────
            # DROP-ONLY: physics-based settling
            elif state == "SETTLE":
                # Pure physics — NO pose writes!
                state_timer += frame_dt
                vel_norm = get_box_vel_norm(current_box_idx)

                settled = (state_timer > 0.3 and vel_norm < args.settle_vel_threshold)
                timed_out = state_timer >= args.settle_s

                if settled or timed_out:
                    final_pos = get_box_pos(current_box_idx)

                    # ── Validate placement ──
                    valid = True
                    reason = ""

                    # Check Z: box bottom near target plane (relaxed threshold)
                    box_bottom = final_pos[2] - 0.5 * dims[2]
                    if box_bottom < PALLET_TOP_Z - 0.10:
                        valid = False
                        reason = (f"Z too low ({box_bottom:.3f} < "
                                  f"{PALLET_TOP_Z - 0.10:.3f})")
                    # Check XY: precise AABB footprint must remain on pallet
                    # (with margin before failing)
                    half_x = 0.5 * dims[0]
                    half_y = 0.5 * dims[1]
                    
                    max_x = 0.5 * PALLET_LX - args.edge_margin
                    max_y = 0.5 * PALLET_LY - args.edge_margin

                    if abs(final_pos[0]) + half_x > max_x:
                        valid = False
                        reason = f"X overhang (pos={final_pos[0]:.3f}, edge={abs(final_pos[0]) + half_x:.3f} > {max_x:.3f})"
                    if abs(final_pos[1]) + half_y > max_y:
                        valid = False
                        reason = f"Y overhang (pos={final_pos[1]:.3f}, edge={abs(final_pos[1]) + half_y:.3f} > {max_y:.3f})"
                    # Check if fell off (Z way too low)
                    if final_pos[2] < 0.0:
                        valid = False
                        reason = f"fell off (z={final_pos[2]:.3f})"

                    # AABB overlap check against all previously placed boxes
                    if valid:
                        for prev_pos, prev_dims in placed_boxes:
                            if aabb_overlap(final_pos, dims, prev_pos, prev_dims,
                                            margin=0.005):
                                valid = False
                                reason = "AABB overlap with placed box"
                                break

                    if valid:
                        # ── Success: freeze box to prevent micro-sliding ──
                        set_kinematic(stage, bp, True)
                        set_disable_gravity(stage, bp, True)
                        zero_box_vel(current_box_idx)

                        placed_boxes.append((final_pos.copy(), dims.copy()))
                        placement_idx += 1
                        current_box_idx += 1
                        retry_count = 0
                        state = "PAUSE"
                        state_timer = 0.0
                        print(f"  ✓ Placed box {placement_idx}/{len(placements)} "
                              f"at ({final_pos[0]:.2f}, {final_pos[1]:.2f}, "
                              f"{final_pos[2]:.2f})  [drop]")
                        # --debug_box_sync: verify the placed box reached sim
                        if args.debug_box_sync:
                            _placed_idx = current_box_idx - 1
                            env.sim.step()
                            env.scene.update(dt=sim_dt)
                            rb = boxes.data.object_pos_w[0, _placed_idx].cpu()
                            target_z = final_pos[2]
                            tag = "OK" if rb[2] > -1.0 else "STUCK at parking z!"
                            print(f"  [BOX_SYNC] box {_placed_idx} "
                                  f"target_z={target_z:.4f}  "
                                  f"readback=({rb[0]:.3f},{rb[1]:.3f},{rb[2]:.3f})  "
                                  f"({tag})")
                            debug_box_sync_prims(stage, box_prim_path(_placed_idx))
                    else:
                        # ── Single-box retry (never full-reset) ──
                        retry_count += 1
                        print(f"  ✗ Placement {placement_idx+1} failed: {reason} "
                              f"(retry {retry_count}/{args.max_retries})")
                        # Park only the failed box
                        set_kinematic(stage, bp, True)
                        set_disable_gravity(stage, bp, True)
                        set_box_pose(current_box_idx, [0, 0, -5], 0.0)

                        if retry_count >= args.max_retries:
                            print(f"  → Skipping placement {placement_idx+1} "
                                  f"after {args.max_retries} retries")
                            placement_idx += 1
                            retry_count = 0

                        current_box_idx += 1
                        state = "SPAWN"
                        state_timer = 0.0

            # ─────────────────────────────────────────
            elif state == "PAUSE":
                # Brief pause to admire the placement
                state_timer += frame_dt
                # ── Patch F: debug frame dump ──
                if args.debug_dump_frame and state_timer < frame_dt * 1.5 and len(frames) > 0:
                    os.makedirs(args.debug_dump_path, exist_ok=True)
                    _dump_path = os.path.join(
                        args.debug_dump_path,
                        f"box_{placement_idx:03d}_frame_{frame_count:06d}.png",
                    )
                    try:
                        import cv2 as _cv2
                        _dump_frame = frames[-1]
                        # In rgb mode, frames are RGB; convert. Otherwise already BGR.
                        if args.record_mode == "rgb":
                            _dump_frame = _cv2.cvtColor(_dump_frame, _cv2.COLOR_RGB2BGR)
                        _cv2.imwrite(_dump_path, _dump_frame)
                        print(f"  [DEBUG] Saved debug frame to {_dump_path}")
                    except Exception as _e:
                        print(f"  [DEBUG] Frame dump failed: {_e}")

                # ── Auto-diagnostic: save 1 RGB + 1 heightmap PNG (once) ──
                if not _diag_saved and placement_idx >= 1 and state_timer < frame_dt * 1.5:
                    _diag_saved = True
                    # RGB diagnostic
                    try:
                        _rgb_diag = env.render()
                        if _rgb_diag is not None:
                            _rgb_path = os.path.join(_diag_dir, "diag_rgb.png")
                            _rgb_bgr = cv2.cvtColor(_rgb_diag, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(_rgb_path, _rgb_bgr)
                            print(f"  [DIAG] Saved diagnostic RGB to {_rgb_path}")
                    except Exception as _e:
                        print(f"  [DIAG] RGB dump failed: {_e}")
                    # Heightmap diagnostic
                    if needs_depth:
                        try:
                            _hmap_bgr = _capture_heightmap_bgr()
                            if _hmap_bgr is not None:
                                _hmap_path = os.path.join(_diag_dir, "diag_heightmap.png")
                                cv2.imwrite(_hmap_path, _hmap_bgr)
                                print(f"  [DIAG] Saved diagnostic heightmap to {_hmap_path}")
                        except Exception as _e:
                            print(f"  [DIAG] Heightmap dump failed: {_e}")

                if state_timer >= 0.15:  # ~4-5 frames at 30fps
                    state = "SPAWN"
                    state_timer = 0.0

        # ── Step physics + render ──
        step_and_capture()
        frame_count += 1

    # ─── Summary ──────────────────────────────────────────────────────
    print(f"\n[INFO] {len(placed_boxes)} boxes placed successfully.")
    print(f"[INFO] Captured {len(frames)} frames.")

    # ─── Write video ──────────────────────────────────────────────────
    print(f"\n[INFO] Writing video to {args.output_path}...")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    if len(frames) > 0:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output_path, fourcc, args.fps, (w, h))
        if not writer.isOpened():
            print(f"[ERROR] Could not open VideoWriter for {args.output_path} "
                  f"(size={w}x{h}, codec=mp4v)")
        else:
            for frame in frames:
                if args.record_mode == "rgb":
                    # RGB frames need conversion to BGR for cv2
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.write(bgr)
                else:
                    # heightmap / both modes already produce BGR frames
                    writer.write(frame)
            writer.release()
        print(f"[SUCCESS] Video: {args.output_path} "
              f"({len(frames)} frames, {len(frames)/args.fps:.1f}s)")
    else:
        print("[WARNING] No frames captured!")

    # ─── Cleanup ──────────────────────────────────────────────────────────
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

# ======================================================================
# Example commands
# ======================================================================
#
# RGB-only (default, unchanged behaviour):
#   python scripts/mockup_video_physics.py --headless \
#       --output_path runs/mockup.mp4
#
# Heightmap-only (agent’s top-down view, inferno colormap, no noise):
#   python scripts/mockup_video_physics.py --headless \
#       --output_path runs/heightmap.mp4 --record_mode heightmap
#
# Side-by-side RGB + heightmap:
#   python scripts/mockup_video_physics.py --headless \
#       --output_path runs/both.mp4 --record_mode both
#
# Custom colormap and range:
#   python scripts/mockup_video_physics.py --headless \
#       --output_path runs/custom.mp4 --record_mode heightmap \
#       --hmap_colormap jet --hmap_vmax 1.5
#
# Heightmap with raw depth dump:
#   python scripts/mockup_video_physics.py --headless \
#       --output_path runs/heightmap.mp4 --record_mode heightmap \
#       --save_depth_raw
