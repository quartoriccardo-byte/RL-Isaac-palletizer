#!/usr/bin/env python3
"""
Physically plausible palletizing mockup video generator.

Key architecture:
  - Pallet PHYSICS is a simple cuboid collider (kinematic, stable).
  - Pallet STL is VISUAL ONLY, auto-aligned by pallet_task.py.
  - Visible floor slab comes from PalletTaskCfg.floor_visual_enabled.
  - Box state machine per placement:
      1) SPAWN:   teleport to side, set kinematic, zero vel
      2) CARRY:   kinematic ease-in-out to above target (no contacts)
      3) LOWER:   kinematic descent at controlled speed
      4) SETTLE:  switch to dynamic, zero vel, pure physics
      5) PAUSE:   brief static view, advance to next box
  - After settle, boxes stay DYNAMIC (natural sleep) by default.
    --freeze_after_settle converts to kinematic (opt-in, may break contacts).
  - Placement validation with AABB overlap check + retry on failure.
  - All pallet_task.py features (floor, mesh, mockup physics) are reused.

Run:
  ~/isaac-sim/python.sh scripts/mockup_video_physics.py \\
      --headless --output_path runs/mockup_physics.mp4

CLI knobs:
  --carry_height, --lower_speed, --release_clearance, --settle_s,
  --freeze_after_settle, --max_retries, --num_boxes, --duration_s, etc.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time


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
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--enable_cameras", action="store_true", default=True)
    parser.add_argument("--cam_width", type=int, default=1280)
    parser.add_argument("--cam_height", type=int, default=720)

    # box counts & seed
    parser.add_argument("--num_boxes", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)

    # motion / stability knobs
    parser.add_argument("--carry_height", type=float, default=1.2,
                        help="Z height during carry phase (m)")
    parser.add_argument("--lower_speed", type=float, default=0.20,
                        help="Kinematic descent speed (m/s)")
    parser.add_argument("--release_clearance", type=float, default=0.04,
                        help="Height above target where box switches to dynamic (m)")
    parser.add_argument("--settle_s", type=float, default=1.5,
                        help="Max settle time per box (seconds)")
    parser.add_argument("--settle_vel_threshold", type=float, default=0.05,
                        help="Velocity norm below which box is considered settled")
    parser.add_argument("--freeze_after_settle", action="store_true", default=False,
                        help="Convert placed box to kinematic after settling "
                             "(opt-in; may break stacking contacts)")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Max retries per box before skipping")
    parser.add_argument("--sim_substeps", type=int, default=4,
                        help="Physics substeps per rendered frame")

    # pallet geometry
    parser.add_argument("--pallet_size_x", type=float, default=1.2)
    parser.add_argument("--pallet_size_y", type=float, default=0.8)
    parser.add_argument("--pallet_thickness", type=float, default=0.15)

    return parser.parse_known_args()


def inject_kit_args(args, unknown):
    """Inject safe Kit/Carb args BEFORE AppLauncher reads sys.argv."""
    if not args.enable_cameras:
        args.enable_cameras = True

    gpu_idx = "0"
    if hasattr(args, "device") and ":" in args.device:
        gpu_idx = args.device.split(":")[-1]

    user_kit_args = [a for a in unknown if a.startswith("--/")]
    user_kit_paths = {a.split("=")[0] for a in user_kit_args}

    defaults = {
        "--/ngx/enabled": "--/ngx/enabled=false",
        "--/rtx/post/dlss/enabled": "--/rtx/post/dlss/enabled=false",
        "--/renderer/multiGpu/enabled": "--/renderer/multiGpu/enabled=false",
        "--/renderer/activeGpu": f"--/renderer/activeGpu={gpu_idx}",
        "--/physics/cudaDevice": f"--/physics/cudaDevice={gpu_idx}",
    }
    for path, arg in defaults.items():
        if path not in user_kit_paths:
            sys.argv.append(arg)
    for arg in user_kit_args:
        if arg not in sys.argv:
            sys.argv.append(arg)


args, unknown = parse_args()
inject_kit_args(args, unknown)


# ═══════════════════════════════════════════════════════════════════════
# 2) Launch Isaac Sim — BEFORE any isaaclab/pxr imports
# ═══════════════════════════════════════════════════════════════════════

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# ═══════════════════════════════════════════════════════════════════════
# 3) Now safe to import Isaac Lab, pxr, torch, etc.
# ═══════════════════════════════════════════════════════════════════════

import torch
import numpy as np

from pxr import UsdPhysics, PhysxSchema

from pallet_rl.envs.pallet_task import PalletTask, PalletTaskCfg


# ═══════════════════════════════════════════════════════════════════════
# USD helpers
# ═══════════════════════════════════════════════════════════════════════

def set_kinematic(stage, prim_path: str, kinematic: bool):
    """Toggle kinematic_enabled on a rigid body via UsdPhysics.RigidBodyAPI."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    rb = UsdPhysics.RigidBodyAPI(prim)
    attr = rb.GetKinematicEnabledAttr()
    if not attr or not attr.IsValid():
        attr = rb.CreateKinematicEnabledAttr()
    attr.Set(bool(kinematic))


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
                         restitution=0.0):
    """Apply/update a UsdPhysics material on a prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    mat_api = UsdPhysics.MaterialAPI.Apply(prim)
    mat_api.CreateStaticFrictionAttr().Set(float(static_friction))
    mat_api.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    mat_api.CreateRestitutionAttr().Set(float(restitution))


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

                # Check overlap with same-layer boxes
                overlap = False
                check_list = layer0_tops if layer_idx == 0 else placements[layer0_count:]
                for prev_pos, prev_dims in (layer0_tops if layer_idx == 0 else
                                             [(p["target_xyz"], p["dims"]) for p in placements[layer0_count:]]):
                    if aabb_overlap(candidate, dims, prev_pos, prev_dims, margin=0.01):
                        overlap = True
                        break
                if not overlap:
                    best_pos = candidate
                    break

            if best_pos is None:
                # Fallback: place at a random position anyway
                tx = float(rng.uniform(-half_lx + margin_x, half_lx - margin_x))
                ty = float(rng.uniform(-half_ly + margin_y, half_ly - margin_y))
                best_pos = np.array([tx, ty, tz], dtype=np.float32)

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

    print(f"\n{'='*60}")
    print(f"  Palletiser Mockup Video — Physics Mode")
    print(f"{'='*60}")
    print(f"  Output:       {args.output_path}")
    print(f"  FPS:          {args.fps}")
    print(f"  Duration:     {args.duration_s}s")
    print(f"  Boxes:        {args.num_boxes}")
    print(f"  Seed:         {args.seed}")
    print(f"  Substeps:     {args.sim_substeps}")
    print(f"  Lower speed:  {args.lower_speed} m/s")
    print(f"  Release gap:  {args.release_clearance} m")
    print(f"  Settle time:  {args.settle_s} s")
    print(f"  Freeze:       {args.freeze_after_settle}")
    print(f"{'='*60}\n")

    # ─── Environment config ───────────────────────────────────────────
    cfg = PalletTaskCfg()
    cfg.scene.num_envs = 1
    cfg.sim.render_interval = 1
    cfg.sim.device = device
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

    stage = omni.usd.get_context().get_stage()
    boxes = env.scene["boxes"]

    # ─── Tune all box prims for stable contacts ───────────────────────
    for i in range(cfg.max_boxes):
        bp = f"/World/envs/env_0/Boxes/box_{i}"
        if stage.GetPrimAtPath(bp).IsValid():
            tune_rigid_body(stage, bp,
                            lin_damp=2.0, ang_damp=3.0,
                            max_depen_vel=0.5, max_lin_vel=2.0,
                            pos_iters=16, vel_iters=4)
            set_physics_material(stage, bp,
                                 static_friction=1.2,
                                 dynamic_friction=0.9,
                                 restitution=0.0)

    # ─── Plan placements ──────────────────────────────────────────────
    placements = generate_placements(
        args.num_boxes, PALLET_LX, PALLET_LY, PALLET_TOP_Z, seed=args.seed
    )
    print(f"[INFO] Planned {len(placements)} placements")

    # ─── Helpers: pose & velocity ─────────────────────────────────────
    def set_box_pose(idx: int, pos_xyz, yaw_rad: float):
        """Write pose for box `idx` in env 0 using torch tensors."""
        pos_t = torch.tensor(pos_xyz, dtype=torch.float32, device=device)
        cy = math.cos(0.5 * yaw_rad)
        sy = math.sin(0.5 * yaw_rad)
        quat_t = torch.tensor([cy, 0.0, 0.0, sy], dtype=torch.float32, device=device)

        boxes.data.object_pos_w[0, idx] = pos_t
        boxes.data.object_quat_w[0, idx] = quat_t
        boxes.data.object_lin_vel_w[0, idx] = 0.0
        boxes.data.object_ang_vel_w[0, idx] = 0.0

        # Write full buffers to sim
        all_pos = boxes.data.object_pos_w.reshape(-1, 3)
        all_quat = boxes.data.object_quat_w.reshape(-1, 4)
        boxes.write_object_pose_to_sim(
            torch.cat([all_pos, all_quat], dim=-1)
        )
        all_lin = boxes.data.object_lin_vel_w.reshape(-1, 3)
        all_ang = boxes.data.object_ang_vel_w.reshape(-1, 3)
        boxes.write_object_velocity_to_sim(
            torch.cat([all_lin, all_ang], dim=-1)
        )

    def zero_box_vel(idx: int):
        """Zero velocities for box `idx`."""
        boxes.data.object_lin_vel_w[0, idx] = 0.0
        boxes.data.object_ang_vel_w[0, idx] = 0.0
        all_lin = boxes.data.object_lin_vel_w.reshape(-1, 3)
        all_ang = boxes.data.object_ang_vel_w.reshape(-1, 3)
        boxes.write_object_velocity_to_sim(
            torch.cat([all_lin, all_ang], dim=-1)
        )

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

    def step_and_capture():
        """Run physics substeps, update scene, render one frame."""
        for _ in range(substeps):
            env.sim.step()
        env.scene.update(dt=sim_dt * substeps)
        fr = env.render()
        if fr is not None:
            frames.append(fr)

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

    print(f"\n[INFO] Starting animation: {total_frames} frames...")

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
                release_z = float(target[2]) + args.release_clearance
                new_z = max(release_z,
                            cur_pos[2] - args.lower_speed * frame_dt)
                pos = [float(target[0]), float(target[1]), new_z]
                set_box_pose(current_box_idx, pos, yaw)

                if abs(new_z - release_z) < 1e-4:
                    # ── Release: switch to dynamic, let physics handle it ──
                    set_kinematic(stage, bp, False)
                    set_disable_gravity(stage, bp, False)
                    zero_box_vel(current_box_idx)
                    state = "SETTLE"
                    state_timer = 0.0

            # ─────────────────────────────────────────
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

                    # Check Z: box bottom should be near target plane
                    box_bottom = final_pos[2] - 0.5 * dims[2]
                    if box_bottom < PALLET_TOP_Z - 0.03:
                        valid = False
                        reason = f"Z too low ({box_bottom:.3f} < {PALLET_TOP_Z - 0.03:.3f})"
                    # Check XY: center within pallet bounds (with tolerance)
                    if abs(final_pos[0]) > 0.5 * PALLET_LX + 0.05:
                        valid = False
                        reason = f"X out of bounds ({final_pos[0]:.3f})"
                    if abs(final_pos[1]) > 0.5 * PALLET_LY + 0.05:
                        valid = False
                        reason = f"Y out of bounds ({final_pos[1]:.3f})"
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
                        # ── Success: keep box dynamic (let it sleep naturally) ──
                        if args.freeze_after_settle:
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
                              f"{final_pos[2]:.2f})")
                    else:
                        # ── Failed — teleport box away and retry ──
                        retry_count += 1
                        print(f"  ✗ Placement {placement_idx+1} failed: {reason} "
                              f"(retry {retry_count}/{args.max_retries})")
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

    try:
        import cv2
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

        if len(frames) > 0:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.output_path, fourcc, args.fps, (w, h))

            for frame in frames:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
            writer.release()
            print(f"[SUCCESS] Video: {args.output_path} "
                  f"({len(frames)} frames, {len(frames)/args.fps:.1f}s)")
        else:
            print("[WARNING] No frames captured!")
    except ImportError:
        print("[ERROR] cv2 not available. Install: pip install opencv-python")

    # ─── Cleanup ──────────────────────────────────────────────────────
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
