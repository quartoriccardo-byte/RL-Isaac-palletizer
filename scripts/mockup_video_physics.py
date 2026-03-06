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
import logging
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
                        choices=["rgb", "heightmap", "both", "diagnostic"],
                        help="What to record: 'rgb' (default, existing behaviour), "
                             "'heightmap' (agent top-down heightmap), "
                             "'both' (side-by-side RGB + heightmap), "
                             "'diagnostic' (composite video + optional raw dumps)")

    # ── diagnostic and raw dumps ─────────────────────────────────────────
    parser.add_argument("--diag_dir", type=str, default="",
                        help="Root directory for diagnostics (default: <output_path>_diagnostics)")
    parser.add_argument("--log_file", type=str, default="run.log",
                        help="Log filename inside diag_dir")
    parser.add_argument("--diag_every_n_frames", type=int, default=50,
                        help="Interval for logging stats and saving raw/vis arrays")
    parser.add_argument("--save_diag_frames", action="store_true",
                        help="Save full composite frames as PNGs at diagnostic interval")
    parser.add_argument("--save_depth_raw", action="store_true",
                        help="Save raw depth frames as .npy at diagnostic interval")
    parser.add_argument("--save_heightmap_raw", action="store_true",
                        help="Save raw heightmap frames as .npy at diagnostic interval")
    parser.add_argument("--save_depth_vis", action="store_true",
                        help="Save depth visualization PNGs at diagnostic interval")
    parser.add_argument("--save_heightmap_vis", action="store_true",
                        help="Save heightmap visualization PNGs at diagnostic interval")

    # ── debug: box sync verification ──────────────────────────────────
    parser.add_argument("--debug_box_sync", action="store_true", default=False,
                        help="After each placement, step once and print the "
                             "box world z to verify it left the parking pose")

    # extensions
    parser.add_argument("--exclude_isaaclab_tasks", action="store_true", default=True,
                        help="Exclude problematic isaaclab_tasks extension")

    # physics fallback & physx gpu buffer sizes
    parser.add_argument("--physics_device", type=str, default="cpu", choices=["cuda", "cpu"],
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
    # /physics/cudaDevice   is NOT set when physics runs on CPU.
    vulkan_idx = "2"   # RTX 6000 in Vulkan device list

    user_kit_args = [a for a in unknown if a.startswith("--/")]
    user_kit_paths = {a.split("=")[0] for a in user_kit_args}

    # Strip any user-supplied --/physics/cudaDevice to prevent PhysX GPU init
    user_kit_args = [a for a in user_kit_args
                     if not a.startswith("--/physics/cudaDevice")]

    defaults = {
        "--/ngx/enabled": "--/ngx/enabled=false",
        "--/rtx/post/dlss/enabled": "--/rtx/post/dlss/enabled=false",
        "--/renderer/multiGpu/enabled": "--/renderer/multiGpu/enabled=false",
        "--/renderer/activeGpu": f"--/renderer/activeGpu={vulkan_idx}",
    }

    # Force CPU physics via Kit setting (prevents PhysX CUDA context creation)
    if args.physics_device == "cpu":
        defaults["--/physics/simulationDevice"] = "--/physics/simulationDevice=cpu"
    else:
        # GPU physics: route PhysX to RTX 6000 (CUDA idx 0)
        defaults["--/physics/cudaDevice"] = "--/physics/cudaDevice=0"
    
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
if args.physics_device == "cpu":
    print(f"[INFO] Kit renderer: Vulkan GPU 2 | PhysX: CPU (no CUDA context)")
else:
    print(f"[INFO] Kit renderer: Vulkan GPU 2 | PhysX CUDA device: 0")

inject_kit_args(args, unknown)


# ═══════════════════════════════════════════════════════════════════════
# 2) Launch Isaac Sim — BEFORE any isaaclab/pxr imports
# ═══════════════════════════════════════════════════════════════════════

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Force Kit settings AFTER AppLauncher, BEFORE env creation ──────────
# AppLauncher may overwrite settings during init.  We re-apply the correct
# values: renderer=Vulkan idx 2, physics=CPU (or GPU if opted in).
try:
    import carb.settings
    _s = carb.settings.get_settings()
    _s.set("/renderer/activeGpu", 2)       # Vulkan ordinal (RTX 6000)
    _s.set("/renderer/multiGpu/enabled", False)
    if args.physics_device == "cpu":
        _s.set("/physics/simulationDevice", "cpu")
        print("[INFO] Post-launch Kit settings forced: "
              "activeGpu=2 (Vulkan), simulationDevice=cpu, multiGpu=false")
    else:
        _s.set("/physics/cudaDevice", 0)   # CUDA ordinal (RTX 6000)
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


def depth_to_bgr(
    depth_m: np.ndarray,
    vmin: float,
    vmax: float,
    colormap_id: int,
    invert: bool = False,
) -> np.ndarray:
    """Convert a (H, W) depth map in meters to (H, W, 3) uint8 BGR.
    Invalid depth readings (<= 0.01) are mapped to black.
    """
    valid_mask = depth_m > 0.01
    d = np.clip(depth_m.astype(np.float64), vmin, vmax)
    norm = ((d - vmin) / max(vmax - vmin, 1e-8) * 255.0).astype(np.uint8)
    if invert:
        norm = 255 - norm
    
    bgr = cv2.applyColorMap(norm, colormap_id)
    # Mask invalid pixels as black
    bgr[~valid_mask] = [0, 0, 0]
    return bgr


def setup_logging(log_path: str, debug: bool) -> logging.Logger:
    """Setup structured file logging for diagnostics."""
    logger = logging.getLogger("mockup_diag")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False
    
    # Avoid duplicate handlers if called multiple times
    if not logger.handlers:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        
        # File handler
        fh = logging.FileHandler(log_path, mode='w')
        fh.setLevel(logging.DEBUG if debug else logging.INFO)
        fh_fmt = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(fh_fmt)
        logger.addHandler(fh)
        
        # Console handler (errors/warnings only to stdout, info goes to file to avoid spam)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.WARNING) 
        ch_fmt = logging.Formatter('%(levelname)s [DIAG]: %(message)s')
        ch.setFormatter(ch_fmt)
        logger.addHandler(ch)
        
    return logger


# ═══════════════════════════════════════════════════════════════════════
# USD helpers — imported from shared utility module
# ═══════════════════════════════════════════════════════════════════════

from pallet_rl.utils.usd_helpers import (
    debug_box_sync_prims,
    set_kinematic,
    set_disable_gravity,
    tune_rigid_body,
    set_physics_material,
    get_render_mesh_prim,
    aabb_overlap,
    aabb_intersection_area,
)



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

    # ─── Device configuration ─────────────────────────────────────────
    # CRITICAL: cfg.sim.device MUST remain on CUDA so that DirectRLEnv,
    # SimulationContext, and all tensors (including Warp heightmaps) stay
    # on the GPU.  GPU PhysX is disabled separately via the Kit-level
    # setting /physics/simulationDevice=cpu (injected by inject_kit_args).
    cfg.sim.device = device   # always CUDA (e.g. "cuda:0" or "cuda:2")
    
    if args.physics_device == "cpu":
        # Zero GPU buffer sizes → PhysX allocates no GPU memory.
        # Broadphase/narrowphase run on CPU via /physics/simulationDevice=cpu.
        cfg.sim.physx.gpu_found_lost_pairs_capacity = 0
        cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 0
        cfg.sim.physx.gpu_heap_capacity = 0
        cfg.sim.physx.gpu_temp_buffer_capacity = 0
        print(f"[INFO] Physics: CPU (Kit simulationDevice=cpu)")
        print(f"[INFO] Tensors/Warp/Env: {device}")
        print("[INFO] PhysX GPU buffers zeroed (no GPU dynamics)")
    else:
        # GPU buffers do not grow dynamically and can crash under heavy contact.
        # Increasing capacities manually prevents PhysX from overflowing its buffers.
        cfg.sim.physx.gpu_found_lost_pairs_capacity = args.gpu_found_lost_pairs_capacity
        cfg.sim.physx.gpu_total_aggregate_pairs_capacity = args.gpu_total_aggregate_pairs_capacity
        cfg.sim.physx.gpu_heap_capacity = args.gpu_heap_capacity
        cfg.sim.physx.gpu_temp_buffer_capacity = args.gpu_temp_buffer_capacity
        print(f"[INFO] Physics: GPU ({device})")
    # NOTE: cfg.max_boxes stays at default (50) — PalletSceneCfg always
    # spawns that many prims and PhysX views have fixed tensor sizes.
    # num_boxes controls how many boxes are "active" for placement.
    cfg.num_boxes = args.num_boxes
    print(f"[INFO] max_boxes={cfg.max_boxes} (fixed), num_boxes={cfg.num_boxes} (active)")
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

    boxes = env.scene["boxes"]
    _warmup_dt = env.sim.get_physics_dt()

    # FIX(B): Park ALL boxes to safe, SEPARATED positions BEFORE any sim step.
    # Default init_state is pos=(0,0,1.5) — all 50 boxes overlap on the pallet,
    # causing PhysX GPU narrowphase overflow (error 700).
    # CRITICAL: Each box must have a UNIQUE (x,y) to avoid interpenetration.
    #   Active boxes   [0..K):  near origin, spread on a 1m grid at z=-5
    #   Inactive boxes [K..M):  far away (base 100,100), 2m grid at z=-5
    K = cfg.num_boxes   # active
    M = cfg.max_boxes   # total prims (always 50)
    _env_ids_pre = torch.tensor([0], dtype=torch.long, device=device)
    print(f"[INFO] Parking {M} boxes (K={K} active, {M-K} inactive) BEFORE warmup ...")

    for _pi in range(M):
        if _pi < K:
            # Active boxes: park near origin on a small grid
            _cols = max(5, int(K**0.5) + 1)
            _px = -3.0 + (_pi % _cols) * 1.0
            _py = -3.0 + (_pi // _cols) * 1.0
        else:
            # Inactive boxes: far-away grid to avoid any scene interaction
            _j = _pi - K
            _cols_inact = 10
            _px = 100.0 + (_j % _cols_inact) * 2.0
            _py = 100.0 + (_j // _cols_inact) * 2.0
        _pz = -5.0

        _st = torch.zeros(1, 1, 13, dtype=torch.float32, device=device)
        _st[0, 0, 0] = _px
        _st[0, 0, 1] = _py
        _st[0, 0, 2] = _pz
        _st[0, 0, 3] = 1.0   # qw (identity)
        _obj_ids = torch.tensor([_pi], dtype=torch.long, device=device)
        boxes.write_object_state_to_sim(_st, _env_ids_pre, _obj_ids)

    # FIX(E): Flush parking writes into PhysX with one sim step
    env.sim.step()
    env.scene.update(dt=_warmup_dt)
    simulation_app.update()

    # FIX(C): Position sanity check — readback and verify
    _all_pos = boxes.data.object_pos_w[0, :M].cpu()  # (M, 3)
    _pos_finite = torch.isfinite(_all_pos).all()
    _pos_absmax = _all_pos.abs().max().item()

    # Active subset diagnostics
    _act = _all_pos[:K]
    print(f"[INFO] Post-park ACTIVE [0..{K}):  "
          f"x=[{_act[:, 0].min():.1f}, {_act[:, 0].max():.1f}]  "
          f"y=[{_act[:, 1].min():.1f}, {_act[:, 1].max():.1f}]  "
          f"z=[{_act[:, 2].min():.1f}, {_act[:, 2].max():.1f}]")

    # Inactive subset diagnostics
    if K < M:
        _inact = _all_pos[K:]
        print(f"[INFO] Post-park INACTIVE [{K}..{M}):  "
              f"x=[{_inact[:, 0].min():.1f}, {_inact[:, 0].max():.1f}]  "
              f"y=[{_inact[:, 1].min():.1f}, {_inact[:, 1].max():.1f}]  "
              f"z=[{_inact[:, 2].min():.1f}, {_inact[:, 2].max():.1f}]")

    print(f"[INFO] Post-park ALL: finite={_pos_finite.item()}, absmax={_pos_absmax:.2f}")
    if not _pos_finite:
        raise RuntimeError(
            f"Box positions contain NaN/Inf after parking! "
            f"Sample: {_all_pos[:min(5, M)].tolist()}"
        )

    # Warmup: step + update a few times so that PhysX GPU broadphase,
    # render products, and depth-camera buffers are all fully active.
    # Boxes are already parked at z=-5 (kinematic), so no contact explosion.
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

    # boxes already assigned above (before parking)

    # ─── Physics / Renderer backend banner ──────────────────────────────
    try:
        import carb.settings
        _settings = carb.settings.get_settings()
        _render_gpu = _settings.get("/renderer/activeGpu")
        _sim_device = _settings.get("/physics/simulationDevice")
        _physics_gpu = _settings.get("/physics/cudaDevice")
        print("")
        print("═" * 52)
        if args.physics_device == "cpu" or _sim_device == "cpu":
            print("  Physics backend : CPU")
            print("  GPU dynamics    : OFF")
        else:
            print(f"  Physics backend : GPU (CUDA device {_physics_gpu})")
            print("  GPU dynamics    : ON")
        print(f"  Tensor/Obs dev  : {device}")
        print(f"  Renderer        : Vulkan, activeGpu = {_render_gpu}")
        print("═" * 52)
        print("")
    except Exception as _e:
        print(f"[WARN] Could not read GPU/physics settings: {_e}")

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
        # Visual properties MUST target the renderable Gprim mesh,
        # NOT the Xform root — otherwise boxes are invisible.
        _mesh_prim = get_render_mesh_prim(stage, box_prim)
        if _mesh_prim is not None:
            UsdShade.MaterialBindingAPI.Apply(_mesh_prim).Bind(_debug_mat)
            # ── Patch V: displayColor/Opacity fallback (Storm ignores unresolved materials) ──
            _gprim = UsdGeom.Gprim(_mesh_prim)
            if _gprim:
                _gprim.CreateDisplayColorAttr().Set(
                    [Gf.Vec3f(0.85, 0.25, 0.20)]   # bright red
                )
                _gprim.CreateDisplayOpacityAttr().Set([1.0])
        else:
            # Fallback: apply to root (should not happen with Isaac Lab assets)
            UsdShade.MaterialBindingAPI.Apply(box_prim).Bind(_debug_mat)
            print(f"[WARN] No Gprim mesh found under {bp}, binding material to Xform root")

    print("[INFO] CCD disabled for all box prims (stability mode)")
    print(f"[INFO] Applied debug material + displayColor to {cfg.max_boxes} box meshes")

    # ─── Debug diagnostics ─────────────────────────────────────────────
    if args.debug or args.debug_box_sync:
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
                # Show resolved mesh prim path for render debugging
                _mesh = get_render_mesh_prim(stage, _dprim)
                if _mesh is not None:
                    _mesh_path = _mesh.GetPath().pathString
                    _is_gprim = bool(UsdGeom.Gprim(_mesh))
                    print(f"           render_mesh = {_mesh_path}  Gprim={_is_gprim}")
                else:
                    print(f"           render_mesh = NONE (⚠ box will be invisible!)")
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
            # Flush PhysX → USD prim transforms
            simulation_app.update()

            rb_pos = boxes.data.object_pos_w[0, idx].cpu()
            all_z = boxes.data.object_pos_w[0, :args.num_boxes, 2].cpu()

            # === USD prim transform readback (renderer's view) ===
            _bp = box_prim_path(idx)
            _prim = stage.GetPrimAtPath(_bp)
            _usd_z = float("nan")
            if _prim.IsValid():
                _xf = UsdGeom.Xformable(_prim)
                _world_mtx = _xf.ComputeLocalToWorldTransform(0.0)
                _usd_pos = _world_mtx.ExtractTranslation()
                _usd_z = _usd_pos[2]

            _match = "OK" if abs(rb_pos[2].item() - _usd_z) < 0.01 else "MISMATCH!"
            print(f"  [BOX_SYNC] set_box_pose({idx}) "
                  f"target=({px:.3f},{py:.3f},{pz:.3f})  "
                  f"tensor_z={rb_pos[2]:.3f}  "
                  f"usd_z={_usd_z:.3f}  [{_match}]  "
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

    # ─── Setup Diagnostic Logging ────────────────────────────────────
    diag_dir = args.diag_dir if args.diag_dir else os.path.join(os.path.dirname(args.output_path) or ".", "mockup_diagnostics")
    if args.record_mode == "diagnostic":
        os.makedirs(diag_dir, exist_ok=True)
        log_path = os.path.join(diag_dir, args.log_file)
        diag_logger = setup_logging(log_path, args.debug)
        diag_logger.info("Started mockup diagnostic recording")
        diag_logger.info(f"Target FPS: {args.fps}, Duration: {args.duration_s}s, Mode: {placement_mode}")
    else:
        diag_logger = None

    # ─── Sim stepping + capture ───────────────────────────────────────
    sim_dt = env.sim.get_physics_dt()
    substeps = args.sim_substeps
    frame_dt = 1.0 / args.fps

    video_writer = None
    depth_frame_idx = 0  # counter for frames

    # Directories for raw and vis dumps
    raw_depth_dir: str | None = None
    if needs_depth and args.save_depth_raw:
        if args.record_mode == "diagnostic":
            raw_depth_dir = os.path.join(diag_dir, "depth_raw")
        else:
            raw_depth_dir = args.depth_raw_dir if hasattr(args, 'depth_raw_dir') and args.depth_raw_dir else os.path.splitext(args.output_path)[0] + "_depth_raw"
        os.makedirs(raw_depth_dir, exist_ok=True)
        print(f"[INFO] Raw depth frames will be saved to: {raw_depth_dir}")
        
    raw_hmap_dir: str | None = None
    if needs_depth and getattr(args, 'save_heightmap_raw', False):
        raw_hmap_dir = os.path.join(diag_dir, "heightmap_raw")
        os.makedirs(raw_hmap_dir, exist_ok=True)
        
    vis_depth_dir: str | None = None
    if needs_depth and getattr(args, 'save_depth_vis', False):
        vis_depth_dir = os.path.join(diag_dir, "depth_vis")
        os.makedirs(vis_depth_dir, exist_ok=True)

    vis_hmap_dir: str | None = None
    if needs_depth and getattr(args, 'save_heightmap_vis', False):
        vis_hmap_dir = os.path.join(diag_dir, "heightmap_vis")
        os.makedirs(vis_hmap_dir, exist_ok=True)

    diag_frames_dir: str | None = None
    if args.record_mode == "diagnostic" and getattr(args, 'save_diag_frames', False):
        diag_frames_dir = os.path.join(diag_dir, "frames")
        os.makedirs(diag_frames_dir, exist_ok=True)

    # Resolve heightmap visualization parameters
    hmap_vmin = args.hmap_vmin
    hmap_vmax = args.hmap_vmax if args.hmap_vmax > 0 else cfg.max_height
    hmap_cmap_id = resolve_cv2_colormap(args.hmap_colormap)
    hmap_invert = args.hmap_invert

    def _capture_diagnostic_data() -> dict:
        """Read depth camera, log stats, save raw dumps, and create BGR views."""
        nonlocal depth_frame_idx
        
        # CRITICAL: explicitly update depth camera so it reads fresh data
        depth_cam.update(dt=sim_dt * substeps)
        
        depth_raw = depth_cam.data.output["distance_to_image_plane"]
        if hasattr(depth_raw, 'detach'):
            depth_t = depth_raw.detach()
        else:
            depth_t = torch.as_tensor(depth_raw, device=device)
        if depth_t.dim() == 4 and depth_t.shape[-1] == 1:
            depth_t = depth_t.squeeze(-1)
        if depth_t.dim() == 2:
            depth_t = depth_t.unsqueeze(0)
            
        is_diag_tick = (depth_frame_idx % args.diag_every_n_frames == 0)
        
        # ── Depth diagnostics ──
        if is_diag_tick:
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
                
            msg = (f"Frame {depth_frame_idx:05d} | Depth Valid: {_valid_count}/{_total} "
                   f"({100*_valid_count/_total:.1f}%) | "
                   f"Min: {_d_min:.3f}m | Max: {_d_max:.3f}m | Mean: {_d_mean:.3f}m")
            if diag_logger:
                diag_logger.info(msg)
            else:
                print(f"  [DEPTH DBG] {msg}")
                
        # Camera pose
        cam_pos = depth_cam.data.pos_w       # (N, 3)
        cam_quat = depth_cam.data.quat_w_world  # (N, 4) wxyz

        if is_diag_tick and diag_logger:
            _cx, _cy, _cz = cam_pos[0].cpu().numpy()
            diag_logger.debug(f"Frame {depth_frame_idx:05d} | Cam Pos (world): ({_cx:.3f}, {_cy:.3f}, {_cz:.3f})")

        # Depth → heightmap via converter
        hmap_t = hmap_converter.depth_to_heightmap(depth_t, cam_pos, cam_quat)
        
        # Heightmap diagnostics
        if is_diag_tick and diag_logger:
            _h0 = hmap_t[0].cpu().numpy()
            _valid_h_mask = np.isfinite(_h0)
            _valid_h_count = int(_valid_h_mask.sum())
            _t_h = _h0.size
            if _valid_h_count > 0:
                _h_min = float(_h0[_valid_h_mask].min())
                _h_max = float(_h0[_valid_h_mask].max())
                _h_mean = float(_h0[_valid_h_mask].mean())
            else:
                _h_min = _h_max = _h_mean = 0.0
            diag_logger.info(
                   f"Frame {depth_frame_idx:05d} | HMap  Valid: {_valid_h_count}/{_t_h} "
                   f"({100*_valid_h_count/_t_h:.1f}%) | "
                   f"Min: {_h_min:.3f}m | Max: {_h_max:.3f}m | Mean: {_h_mean:.3f}m")

        # Visualizations
        depth_np = depth_t[0].cpu().numpy()
        hmap_np = hmap_t[0].cpu().numpy()
        
        depth_bgr = depth_to_bgr(depth_np, 0.0, float(cfg.depth_cam_height_m), hmap_cmap_id, hmap_invert)
        hmap_bgr = heightmap_to_bgr(hmap_np, hmap_vmin, hmap_vmax, hmap_cmap_id, hmap_invert)

        # Raw & Vis saving at interval
        if is_diag_tick:
            if raw_depth_dir:
                np.save(os.path.join(raw_depth_dir, f"depth_{depth_frame_idx:06d}.npy"), depth_np.astype(np.float32))
            if raw_hmap_dir:
                np.save(os.path.join(raw_hmap_dir, f"hmap_{depth_frame_idx:06d}.npy"), hmap_np.astype(np.float32))
            if vis_depth_dir:
                cv2.imwrite(os.path.join(vis_depth_dir, f"depth_vis_{depth_frame_idx:06d}.png"), depth_bgr)
            if vis_hmap_dir:
                cv2.imwrite(os.path.join(vis_hmap_dir, f"hmap_vis_{depth_frame_idx:06d}.png"), hmap_bgr)

        depth_frame_idx += 1
        
        return {
            "depth_bgr": depth_bgr,
            "hmap_bgr": hmap_bgr,
        }

    def _capture_heightmap_bgr() -> np.ndarray | None:
        """Legacy helper for non-diagnostic mode."""
        return _capture_diagnostic_data()["hmap_bgr"]

    def step_and_capture():
        nonlocal video_writer
        
        for _ in range(substeps):
            env.sim.step()
        env.scene.update(dt=sim_dt * substeps)

        # CRITICAL: Flush USD transforms into the renderer.
        simulation_app.update()

        record_mode = args.record_mode
        frame_to_write = None

        if record_mode == "rgb":
            frame_to_write = env.render()
            if frame_to_write is not None:
                frame_to_write = cv2.cvtColor(frame_to_write, cv2.COLOR_RGB2BGR)

        elif record_mode == "heightmap":
            frame_to_write = _capture_heightmap_bgr()

        elif record_mode == "both":
            rgb_frame = env.render()
            hmap_bgr = _capture_heightmap_bgr()
            if rgb_frame is not None and hmap_bgr is not None:
                rgb_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                rgb_h, rgb_w = rgb_bgr.shape[:2]
                hm_h, hm_w = hmap_bgr.shape[:2]
                if hm_h != rgb_h:
                    scale = rgb_h / hm_h
                    new_w = int(hm_w * scale)
                    hmap_bgr = cv2.resize(hmap_bgr, (new_w, rgb_h), interpolation=cv2.INTER_LINEAR)
                frame_to_write = np.concatenate([rgb_bgr, hmap_bgr], axis=1)
            elif rgb_frame is not None:
                rgb_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                rgb_h, rgb_w = rgb_bgr.shape[:2]
                raw_hm_h, raw_hm_w = cfg.map_shape
                scale = rgb_h / raw_hm_h
                placeholder_w = int(raw_hm_w * scale)
                placeholder = np.zeros((rgb_h, placeholder_w, 3), dtype=np.uint8)
                frame_to_write = np.concatenate([rgb_bgr, placeholder], axis=1)
                
        elif record_mode == "diagnostic":
            rgb_frame = env.render()
            diag_data = _capture_diagnostic_data()
            depth_bgr = diag_data["depth_bgr"]
            hmap_bgr = diag_data["hmap_bgr"]
            
            if rgb_frame is not None:
                rgb_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                rgb_h, rgb_w = rgb_bgr.shape[:2]
                
                # Scale depth and heightmap to match RGB height
                for bgr in (depth_bgr, hmap_bgr):
                    h, w = bgr.shape[:2]
                    if h != rgb_h:
                        bgr_resized = cv2.resize(bgr, (int(w * (rgb_h / h)), rgb_h), interpolation=cv2.INTER_LINEAR)
                        if bgr is depth_bgr: depth_bgr = bgr_resized
                        if bgr is hmap_bgr: hmap_bgr = bgr_resized
                        
                # Add text labels (they will be written on the BGR arrays)
                cv2.putText(rgb_bgr, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(depth_bgr, "Depth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(hmap_bgr, "Heightmap", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                frame_to_write = np.concatenate([rgb_bgr, depth_bgr, hmap_bgr], axis=1)
                
                if diag_frames_dir and (depth_frame_idx - 1) % args.diag_every_n_frames == 0:
                    cv2.imwrite(os.path.join(diag_frames_dir, f"diag_frame_{depth_frame_idx-1:06d}.png"), frame_to_write)

        if frame_to_write is not None:
            if video_writer is None:
                h, w = frame_to_write.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
                video_writer = cv2.VideoWriter(args.output_path, fourcc, args.fps, (w, h))
                print(f"[INFO] Initialized VideoWriter: {w}x{h} @ {args.fps}fps -> {args.output_path}")
                if not video_writer.isOpened():
                    print(f"[ERROR] Failed to open VideoWriter for {args.output_path}")
            
            if video_writer.isOpened():
                video_writer.write(frame_to_write)
    # ─── Park all boxes + set kinematic (USD-level) ────────────────────
    # NOTE(B): Pose parking was already done BEFORE warmup (via tensor API).
    # Here we additionally set kinematic + disable gravity via USD APIs,
    # which must happen after `stage` is available.
    for i in range(cfg.max_boxes):
        bp = box_prim_path(i)
        set_kinematic(stage, bp, True)
        set_disable_gravity(stage, bp, True)

    # FIX(G): Diagnostic — verify positions after full parking
    if args.debug or args.debug_box_sync:
        _post_park_pos = boxes.data.object_pos_w[0, :cfg.max_boxes, 2].cpu()
        print(f"  [PARK DBG] all_z after parking: "
              f"min={_post_park_pos.min():.2f} max={_post_park_pos.max():.2f}")

    # Let the scene settle for a few frames
    for _ in range(10):
        step_and_capture()

    # ─── CRITICAL: Re-orient cameras AFTER warmup + parking ───────────
    # The camera look-at set in PalletTask.__init__() may not persist
    # after warmup/parking steps.  The legacy mockup_video.py explicitly
    # re-sets the camera AFTER warmup — we do the same here.
    try:
        render_cam = env.scene["render_camera"]
        _cam_eye = torch.tensor([[2.5, 2.5, 2.5]], dtype=torch.float32,
                                device=device)
        _cam_target = torch.tensor([[0.0, 0.0, 0.3]], dtype=torch.float32,
                                   device=device)
        render_cam.set_world_poses_from_view(eyes=_cam_eye, targets=_cam_target)
        print("[CAMERA] Re-oriented render camera after warmup: "
              f"eye=(2.5,2.5,2.5) target=(0,0,0.3)")
    except Exception as _e:
        print(f"[CAMERA WARN] Could not re-orient render camera: {_e}")

    if needs_depth and depth_cam is not None:
        try:
            _dc_eye = torch.tensor([[0.0, 0.0, cfg.depth_cam_height_m]],
                                   dtype=torch.float32, device=device)
            _dc_tgt = torch.tensor([[0.0, 0.0, 0.0]],
                                   dtype=torch.float32, device=device)
            depth_cam.set_world_poses_from_view(eyes=_dc_eye, targets=_dc_tgt)
            print(f"[CAMERA] Re-oriented depth camera after warmup: "
                  f"eye=(0,0,{cfg.depth_cam_height_m}) target=(0,0,0)")
        except Exception as _e:
            print(f"[CAMERA WARN] Could not re-orient depth camera: {_e}")

    # Pump a few more render frames to let the camera re-orientation take effect
    for _ in range(3):
        simulation_app.update()

    # ─── World-frame diagnostics ──────────────────────────────────────
    print("\n[DIAG] === World-Frame Diagnostics ===")

    # Physics backend actually used
    try:
        import carb.settings as _csettings
        _sim_dev  = _csettings.get_settings().get("/physics/simulationDevice")
        _csdev    = _csettings.get_settings().get("/physics/cudaDevice")
        _agpu     = _csettings.get_settings().get("/renderer/activeGpu")
        if _sim_dev == "cpu" or args.physics_device == "cpu":
            print(f"  Physics backend   : CPU")
        else:
            print(f"  PhysX cudaDevice  : {_csdev}  (CUDA ordinal)")
        print(f"  Renderer activeGpu: {_agpu}  (Vulkan ordinal)")
    except Exception as _e:
        print(f"  GPU settings      : [error: {_e}]")

    # Pallet position: tensor readback vs USD prim
    try:
        _pallet = env.scene["pallet"]
        _pallet_pos = _pallet.data.root_pos_w[0].cpu().numpy()
        print(f"  Pallet tensor pos : ({_pallet_pos[0]:.3f}, {_pallet_pos[1]:.3f}, {_pallet_pos[2]:.3f})")
    except Exception as _e:
        print(f"  Pallet tensor pos : [error: {_e}]")
    try:
        _pallet_prim = stage.GetPrimAtPath("/World/envs/env_0/Pallet")
        if _pallet_prim.IsValid():
            _pxf = UsdGeom.Xformable(_pallet_prim)
            _pmtx = _pxf.ComputeLocalToWorldTransform(0.0)
            _ptrans = _pmtx.ExtractTranslation()
            print(f"  Pallet USD pos    : ({_ptrans[0]:.3f}, {_ptrans[1]:.3f}, {_ptrans[2]:.3f})")
        else:
            print(f"  Pallet USD prim   : INVALID")
    except Exception as _e:
        print(f"  Pallet USD pos    : [error: {_e}]")

    # Box 0 position (should be parked at z=-5): tensor vs USD
    try:
        _b0_tensor = boxes.data.object_pos_w[0, 0].cpu().numpy()
        print(f"  Box 0 tensor pos  : ({_b0_tensor[0]:.3f}, {_b0_tensor[1]:.3f}, {_b0_tensor[2]:.3f})")
    except Exception as _e:
        print(f"  Box 0 tensor pos  : [error: {_e}]")
    try:
        _b0_prim = stage.GetPrimAtPath("/World/envs/env_0/Boxes/box_0")
        if _b0_prim.IsValid():
            _b0xf = UsdGeom.Xformable(_b0_prim)
            _b0mtx = _b0xf.ComputeLocalToWorldTransform(0.0)
            _b0trans = _b0mtx.ExtractTranslation()
            print(f"  Box 0 USD pos     : ({_b0trans[0]:.3f}, {_b0trans[1]:.3f}, {_b0trans[2]:.3f})")
        else:
            print(f"  Box 0 USD prim    : INVALID")
    except Exception as _e:
        print(f"  Box 0 USD pos     : [error: {_e}]")

    # Render camera pose
    try:
        _rc_pos = render_cam.data.pos_w[0].cpu().numpy()
        print(f"  Render cam pos    : ({_rc_pos[0]:.3f}, {_rc_pos[1]:.3f}, {_rc_pos[2]:.3f})")
        _rc_quat = render_cam.data.quat_w_world[0].cpu().numpy()
        print(f"  Render cam quat   : ({_rc_quat[0]:.3f}, {_rc_quat[1]:.3f}, {_rc_quat[2]:.3f}, {_rc_quat[3]:.3f})")
    except Exception as _e:
        print(f"  Render cam pose   : [error: {_e}]")

    # Depth camera pose
    if needs_depth and depth_cam is not None:
        try:
            _dc_pos = depth_cam.data.pos_w[0].cpu().numpy()
            _dc_quat = depth_cam.data.quat_w_world[0].cpu().numpy()
            print(f"  Depth cam pos     : ({_dc_pos[0]:.3f}, {_dc_pos[1]:.3f}, {_dc_pos[2]:.3f})")
            print(f"  Depth cam quat    : ({_dc_quat[0]:.3f}, {_dc_quat[1]:.3f}, {_dc_quat[2]:.3f}, {_dc_quat[3]:.3f})")
        except Exception as _e:
            print(f"  Depth cam pose    : [error: {_e}]")

    # Env origin (with num_envs=1, should be (0,0,0))
    try:
        _env_orig = env.scene.env_origins[0].cpu().numpy()
        print(f"  Env 0 origin      : ({_env_orig[0]:.3f}, {_env_orig[1]:.3f}, {_env_orig[2]:.3f})")
    except Exception as _e:
        print(f"  Env 0 origin      : [error: {_e}]")

    print("[DIAG] === End Diagnostics ===\n")

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
        if placement_idx < len(placements) and current_box_idx < cfg.num_boxes:
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

                        # FIX(F): Wrap index to recycle prims within active range
                        current_box_idx = (current_box_idx + 1) % cfg.num_boxes
                        state = "SPAWN"
                        state_timer = 0.0

            # ─────────────────────────────────────────
            elif state == "PAUSE":
                # Brief pause to admire the placement
                state_timer += frame_dt
                # ── Patch F: debug frame dump ──
                if args.debug_dump_frame and state_timer < frame_dt * 1.5:
                    os.makedirs(args.debug_dump_path, exist_ok=True)
                    _dump_path = os.path.join(
                        args.debug_dump_path,
                        f"box_{placement_idx:03d}_frame_{frame_count:06d}.png",
                    )
                    try:
                        import cv2 as _cv2
                        _dump_frame = env.render()
                        if _dump_frame is not None:
                            _dump_bgr = _cv2.cvtColor(_dump_frame, _cv2.COLOR_RGB2BGR)
                            _cv2.imwrite(_dump_path, _dump_bgr)
                            print(f"  [DEBUG] Saved debug frame to {_dump_path}")
                    except Exception as _e:
                        print(f"  [DEBUG] Frame dump failed: {_e}")

                # ── Auto-diagnostic: save 1 RGB + 1 heightmap PNG (once) ──
                if not _diag_saved and placement_idx >= 1 and state_timer < frame_dt * 1.5:
                    _diag_saved = True
                    # Print placed box world position for coordinate frame verification
                    try:
                        _box0_pos = boxes.data.object_pos_w[0, 0].cpu().numpy()
                        _rcam_pos = render_cam.data.pos_w[0].cpu().numpy()
                        print(f"  [DIAG] Box 0 world pos : ({_box0_pos[0]:.3f}, {_box0_pos[1]:.3f}, {_box0_pos[2]:.3f})")
                        print(f"  [DIAG] Render cam pos  : ({_rcam_pos[0]:.3f}, {_rcam_pos[1]:.3f}, {_rcam_pos[2]:.3f})")
                    except Exception:
                        pass
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
    print(f"[INFO] Processed {frame_count} frames.")

    if video_writer is not None:
        video_writer.release()
        print(f"[SUCCESS] Video: {args.output_path} "
              f"({frame_count} frames, {frame_count/args.fps:.1f}s)")
    else:
        print("[WARNING] No frames written to video (VideoWriter not opened)!")

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
