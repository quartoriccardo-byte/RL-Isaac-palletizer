"""
[LEGACY] Mockup Video Generator for Palletizer RL

⚠️  DEPRECATED — Use scripts/mockup_video_physics.py instead.
    This script uses teleport-based animation which causes clip-then-snap artifacts.
    The physics-based version uses a proper kinematic→dynamic state machine.

Generates a ~20-second, 30 FPS demonstration video of plausible box packing
on a Euro pallet. Uses scripted placement heuristics (not a trained policy)
to showcase the environment with floor, pallet mesh, colored boxes, and
visible retry/reset on failed placements.

Usage (legacy):
    ~/isaac-sim/python.sh scripts/mockup_video.py --headless --output_path mockup_demo.mp4
    ~/isaac-sim/python.sh scripts/mockup_video.py --headless --num_boxes 20 --duration_s 25
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch
from pallet_rl.utils.device_utils import pick_supported_cuda_device

# =============================================================================
# CRITICAL: Parse args and launch Isaac BEFORE any sim imports
# =============================================================================


def parse_args():
    """Parse CLI arguments, allowing unknown Kit/Carb settings (--/rtx/...).

    Returns:
        tuple: (args, unknown) where unknown may contain --/path=value args.
    """
    parser = argparse.ArgumentParser(description="Generate palletizer mockup video")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--output_path", type=str, default="mockup_demo.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration_s", type=float, default=20.0)
    parser.add_argument("--num_boxes", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--enable_cameras", action="store_true", default=True)
    parser.add_argument("--cam_width", type=int, default=1280)
    parser.add_argument("--cam_height", type=int, default=720)
    parser.add_argument("--sim_substeps", type=int, default=4,
                        help="Physics substeps per rendered frame (more = smoother)")
    parser.add_argument("--drop_speed_mps", type=float, default=0.15,
                        help="Controlled descent speed in m/s during kinematic lowering")
    parser.add_argument("--release_gap", type=float, default=0.01,
                        help="Gap (m) above resting position where box is released to physics")
    parser.add_argument("--settle_frames", type=int, default=80,
                        help="Number of frames to let physics settle after release (no pose writes)")
    return parser.parse_known_args()


def inject_kit_args(args, unknown):
    """Inject Kit/Carb startup args into sys.argv before AppLauncher init.

    Adds safe defaults (NGX/DLSS disable, VRAM reduction) unless the user
    already provided the same --/path via CLI.  Also forces cameras on
    since the mockup script always needs them for recording.

    GPU selection notes:
      --/renderer/activeGpu=N   selects the Vulkan/RTX rendering GPU.
      --/physics/cudaDevice=N   selects the PhysX CUDA GPU.
      These are NOT controlled by CUDA_VISIBLE_DEVICES.  On multi-GPU
      machines with mixed architectures (e.g. GTX 1080 Ti + RTX 6000),
      both must point at an RTX-capable GPU (compute capability >= 7.0).
    """
    # Force cameras for recording
    if not args.enable_cameras:
        args.enable_cameras = True
        print("[INFO] Forced --enable_cameras for mockup recording")

    # Infer GPU index from --device (e.g. "cuda:2" -> "2", "cuda" -> "0")
    gpu_idx = "0"
    if hasattr(args, "device") and ":" in args.device:
        gpu_idx = args.device.split(":")[-1]

    # Collect user-provided Kit args
    user_kit_args = [arg for arg in unknown if arg.startswith('--/')]
    user_kit_paths = {arg.split('=')[0] for arg in user_kit_args}

    # Safe defaults (only added when user didn't override)
    defaults_map = {
        '--/ngx/enabled': '--/ngx/enabled=false',
        '--/rtx/post/dlss/enabled': '--/rtx/post/dlss/enabled=false',
        '--/rtx/post/dlss/execMode': '--/rtx/post/dlss/execMode=0',
        '--/rtx/post/aa/op': '--/rtx/post/aa/op=0',
        '--/rtx-transient/dlssg/enabled': '--/rtx-transient/dlssg/enabled=false',
        '--/rtx-transient/dldenoiser/enabled': '--/rtx-transient/dldenoiser/enabled=false',
        '--/renderer/multiGpu/enabled': '--/renderer/multiGpu/enabled=false',
        '--/rtx/translucency/enabled': '--/rtx/translucency/enabled=false',
        '--/rtx/reflections/enabled': '--/rtx/reflections/enabled=false',
        '--/rtx/indirectDiffuse/enabled': '--/rtx/indirectDiffuse/enabled=false',
        # GPU pinning: force renderer and PhysX onto the correct GPU
        '--/renderer/activeGpu': f'--/renderer/activeGpu={gpu_idx}',
        '--/physics/cudaDevice': f'--/physics/cudaDevice={gpu_idx}',
    }
    default_kit_args = []
    for kit_path, kit_arg in defaults_map.items():
        if kit_path not in user_kit_paths:
            default_kit_args.append(kit_arg)

    # Inject into sys.argv BEFORE AppLauncher reads them
    for arg in user_kit_args + default_kit_args:
        if arg not in sys.argv:
            sys.argv.append(arg)
            print(f"[INFO] Injected Kit startup arg: {arg}")


args, unknown = parse_args()

# Force supported GPU (RTX 6000 vs 1080 Ti)
_, forced_device = pick_supported_cuda_device()
args.device = forced_device
print(f"[INFO] Overriding CLI device with forced supported GPU: {args.device}")
inject_kit_args(args, unknown)

# Launch Isaac Sim BEFORE any isaaclab imports
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now safe to import Isaac Lab modules
import torch
import numpy as np
import random

from pallet_rl.envs.pallet_task import PalletTask, PalletTaskCfg


# =============================================================================
# Placement Heuristic
# =============================================================================


class BottomLeftFirstPacker:
    """
    Simple bottom-left-first bin packing heuristic for 2D layers.
    
    Maintains an occupancy grid and places boxes in the first available
    position scanning left→right, bottom→top.
    """

    def __init__(
        self,
        pallet_l: float = 1.2,
        pallet_w: float = 0.8,
        grid_res_x: int = 16,
        grid_res_y: int = 24,
        max_height: float = 1.8,
    ):
        self.pallet_l = pallet_l
        self.pallet_w = pallet_w
        self.grid_res_x = grid_res_x
        self.grid_res_y = grid_res_y
        self.max_height = max_height
        self.cell_w = pallet_l / grid_res_x
        self.cell_h = pallet_w / grid_res_y

        # Height grid: track max height at each cell
        self.height_grid = np.zeros((grid_res_x, grid_res_y), dtype=np.float32)

    def find_placement(
        self, box_l: float, box_w: float, box_h: float, try_rotate: bool = True
    ) -> tuple[int, int, int, float] | None:
        """
        Find valid placement for a box.
        
        Returns (grid_x, grid_y, rotation, z_height) or None if no fit.
        rotation: 0 = no rotation, 1 = 90° rotation.
        """
        orientations = [(box_l, box_w, 0)]
        if try_rotate and abs(box_l - box_w) > 0.01:
            orientations.append((box_w, box_l, 1))

        best = None
        best_z = float("inf")

        for bl, bw, rot in orientations:
            # How many cells does the box span?
            cells_x = max(1, int(math.ceil(bl / self.cell_w)))
            cells_y = max(1, int(math.ceil(bw / self.cell_h)))

            for gx in range(self.grid_res_x - cells_x + 1):
                for gy in range(self.grid_res_y - cells_y + 1):
                    region = self.height_grid[gx : gx + cells_x, gy : gy + cells_y]
                    z = float(region.max())

                    if z + box_h > self.max_height:
                        continue

                    # Prefer lower placements
                    if best is None or z < best_z:
                        best = (gx, gy, rot, z)
                        best_z = z

        return best

    def commit_placement(
        self,
        grid_x: int,
        grid_y: int,
        box_l: float,
        box_w: float,
        box_h: float,
        rotation: int,
    ):
        """Mark cells as occupied after successful placement."""
        if rotation == 1:
            box_l, box_w = box_w, box_l

        cells_x = max(1, int(math.ceil(box_l / self.cell_w)))
        cells_y = max(1, int(math.ceil(box_w / self.cell_h)))

        z_base = float(
            self.height_grid[
                grid_x : grid_x + cells_x, grid_y : grid_y + cells_y
            ].max()
        )
        self.height_grid[
            grid_x : grid_x + cells_x, grid_y : grid_y + cells_y
        ] = z_base + box_h

    def grid_to_world(self, gx: int, gy: int, box_l: float, box_w: float, rotation: int) -> tuple[float, float]:
        """Convert grid coordinates to world XY (pallet-centered)."""
        if rotation == 1:
            box_l, box_w = box_w, box_l

        cells_x = max(1, int(math.ceil(box_l / self.cell_w)))
        cells_y = max(1, int(math.ceil(box_w / self.cell_h)))

        # Center of the box in grid space
        cx = (gx + cells_x / 2.0) * self.cell_w - self.pallet_l / 2.0
        cy = (gy + cells_y / 2.0) * self.cell_h - self.pallet_w / 2.0

        return cx, cy


# =============================================================================
# Box Dimension Generator
# =============================================================================


def generate_box_dims(num_boxes: int, seed: int = 42) -> list[tuple[float, float, float]]:
    """
    Generate realistic box dimensions (L, W, H) in meters.
    
    Mix of small, medium, and large boxes typical in logistics.
    """
    rng = random.Random(seed)
    
    # Box templates (L, W, H ranges)
    templates = [
        # Small boxes
        {"l": (0.15, 0.25), "w": (0.10, 0.20), "h": (0.08, 0.15)},
        # Medium boxes
        {"l": (0.25, 0.40), "w": (0.20, 0.30), "h": (0.12, 0.25)},
        # Large boxes
        {"l": (0.35, 0.55), "w": (0.25, 0.40), "h": (0.15, 0.35)},
        # Flat boxes
        {"l": (0.30, 0.50), "w": (0.25, 0.40), "h": (0.05, 0.10)},
    ]
    
    boxes = []
    for i in range(num_boxes):
        t = rng.choice(templates)
        l = round(rng.uniform(*t["l"]), 3)
        w = round(rng.uniform(*t["w"]), 3)
        h = round(rng.uniform(*t["h"]), 3)
        boxes.append((l, w, h))
    
    return boxes


# =============================================================================
# Box Color Palette
# =============================================================================

BOX_COLORS = [
    (0.85, 0.65, 0.40),   # Cardboard brown
    (0.90, 0.72, 0.45),   # Light brown
    (0.75, 0.55, 0.35),   # Dark cardboard
    (0.80, 0.60, 0.42),   # Kraft paper
    (0.88, 0.78, 0.55),   # Sandy
    (0.70, 0.50, 0.30),   # Walnut
    (0.82, 0.68, 0.48),   # Tan
    (0.55, 0.45, 0.35),   # Dark brown (special box)
    (0.30, 0.50, 0.70),   # Blue (marked box)
    (0.70, 0.30, 0.30),   # Red (fragile)
]


# =============================================================================
# Main Mockup Script
# =============================================================================


def main():
    import gymnasium as gym
    from gymnasium.wrappers import RecordVideo
    from isaacsim.core.utils import prims as prim_utils
    
    print(f"\n{'='*60}")
    print(f"  Palletizer Mockup Video Generator")
    print(f"{'='*60}")
    print(f"  Output:     {args.output_path}")
    print(f"  FPS:        {args.fps}")
    print(f"  Duration:   {args.duration_s}s")
    print(f"  Boxes:      {args.num_boxes}")
    print(f"  Seed:       {args.seed}")
    print(f"{'='*60}\n")
    
    device = args.device
    
    # =========================================================================
    # Create environment with mockup-optimized config
    # =========================================================================
    cfg = PalletTaskCfg()
    cfg.scene.num_envs = 1
    cfg.sim.render_interval = 1  # Every frame for video
    cfg.max_boxes = max(args.num_boxes + 5, cfg.max_boxes)  # Extra for retries
    cfg.use_pallet_mesh_visual = True  # Show Euro pallet mesh
    cfg.mockup_mode = True              # Gentle physics for demo video
    cfg.decimation = 1                  # Single physics step per env.step()
    
    # Ensure sim device matches CLI --device (e.g. "cuda:2")
    # Without this, PalletTaskCfg.sim.device="cuda" would not carry the index.
    cfg.sim.device = args.device
    
    # Override camera for cinematic view
    cfg.scene.render_camera.width = args.cam_width
    cfg.scene.render_camera.height = args.cam_height
    
    # Create env
    env = PalletTask(cfg, render_mode="rgb_array")
    
    # Warm up camera (Isaac Lab requires a few steps)
    print("[INFO] Warming up camera...")
    obs, info = env.reset()
    for _ in range(5):
        env.sim.step()
        env.scene.update(dt=env.sim.get_physics_dt())
    
    # Set up cinematic camera look-at
    cam = env.scene["render_camera"]
    eye = torch.tensor([[2.8, 2.2, 2.0]], dtype=torch.float32, device=device)
    target = torch.tensor([[0.0, 0.0, 0.6]], dtype=torch.float32, device=device)
    cam.set_world_poses_from_view(eye, target)
    
    # Wrap with RecordVideo
    output_dir = os.path.dirname(args.output_path) or "."
    output_name = os.path.splitext(os.path.basename(args.output_path))[0]
    
    # We'll collect frames manually and write video at the end
    frames = []
    
    # =========================================================================
    # Helper: multi-substep physics + capture
    # =========================================================================
    sim_dt = env.sim.get_physics_dt()
    substeps = args.sim_substeps
    
    def _step_and_capture(frames_list):
        """Run N physics substeps, update scene, render one frame."""
        for _ in range(substeps):
            env.sim.step()
        env.scene.update(dt=sim_dt * substeps)
        fr = env.render()
        if fr is not None:
            frames_list.append(fr)
    
    # =========================================================================
    # Generate box dimensions and plan placements
    # =========================================================================
    box_dims = generate_box_dims(args.num_boxes, seed=args.seed)
    packer = BottomLeftFirstPacker()
    
    print(f"[INFO] Planning {args.num_boxes} box placements...")
    
    # Plan all placements
    placements = []
    for i, (bl, bw, bh) in enumerate(box_dims):
        result = packer.find_placement(bl, bw, bh)
        if result is None:
            print(f"  Box {i}: no fit ({bl:.3f}×{bw:.3f}×{bh:.3f}), skipping")
            continue
        
        gx, gy, rot, z_base = result
        wx, wy = packer.grid_to_world(gx, gy, bl, bw, rot)
        
        placements.append({
            "box_id": i,
            "dims": (bl, bw, bh),
            "grid": (gx, gy),
            "rotation": rot,
            "world_xy": (wx, wy),
            "z_base": z_base,
        })
        packer.commit_placement(gx, gy, bl, bw, bh, rot)
        
        print(f"  Box {i}: {bl:.3f}×{bw:.3f}×{bh:.3f} → grid({gx},{gy}) rot={rot} z={z_base:.3f}")
    
    print(f"\n[INFO] Planned {len(placements)} placements")
    
    # =========================================================================
    # Intentional "bad" placement for retry demo (swap box 3 if it exists)
    # =========================================================================
    RETRY_BOX_IDX = min(3, len(placements) - 1) if len(placements) > 3 else None
    
    # =========================================================================
    # Animation Loop
    # =========================================================================
    total_frames = int(args.duration_s * args.fps)
    frames_per_box = max(total_frames // max(len(placements), 1), 30)
    
    # Phases per box animation (cinematic state machine):
    # 1. Spawn at side (5 frames) — show box appearing
    # 2. Carry to hover (20 frames — smooth interpolation) 
    # 3. Controlled descent (variable frames — kinematic lowering at drop_speed_mps)
    #    → stops at release_gap above resting position
    # 4. Release + settle (settle_frames — PURE PHYSICS, no pose writes!)
    #    → box falls release_gap under gravity and settles naturally
    # 5. Pause (5 frames — admire placement)
    SPAWN_FRAMES = 5
    CARRY_FRAMES = 20
    PAUSE_FRAMES = 5
    
    # Retry animation extras
    RETRY_BACKUP_FRAMES = 10
    RETRY_REAPPROACH_FRAMES = 15
    
    print(f"\n[INFO] Generating {total_frames} frames ({args.duration_s}s @ {args.fps}fps)...")
    print(f"[INFO] Cinematic params: drop_speed={args.drop_speed_mps} m/s, "
          f"release_gap={args.release_gap}m, settle_frames={args.settle_frames}")
    
    placed_count = 0
    frame_count = 0
    
    for pi, placement in enumerate(placements):
        if frame_count >= total_frames:
            break
        
        bl, bw, bh = placement["dims"]
        wx, wy = placement["world_xy"]
        z_base = placement["z_base"]
        rot = placement["rotation"]
        box_id = placement["box_id"]
        
        # Target resting position (center of box = z_base + half height)
        target_z = z_base + bh / 2.0
        target_pos = torch.tensor([wx, wy, target_z], device=device)
        
        # Spawn position (off to the side, safely above everything)
        spawn_pos = torch.tensor([1.2, -0.5, 0.6], device=device)
        
        # Hover position: above target, high enough to clear any stacked boxes
        hover_pos = target_pos.clone()
        hover_pos[2] = z_base + bh + cfg.mockup_drop_height_m
        
        # Release position: just above resting (the box will fall release_gap under gravity)
        release_z = target_z + args.release_gap
        
        # Rotation quaternion (wxyz format)
        if rot == 0:
            quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        else:
            # 90° around Z
            quat = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=device)
        
        is_retry_box = (RETRY_BOX_IDX is not None and pi == RETRY_BOX_IDX)
        
        # =================================================================
        # Phase 1: Spawn at side (kinematic — set pose each frame)
        # =================================================================
        for f in range(SPAWN_FRAMES):
            if frame_count >= total_frames:
                break
            env._set_box_pose(placed_count, spawn_pos, quat, device)
            _step_and_capture(frames)
            frame_count += 1
        
        # =================================================================
        # Phase 2: Carry from spawn to hover (kinematic smooth interp)
        # =================================================================
        for f in range(CARRY_FRAMES):
            if frame_count >= total_frames:
                break
            t = f / max(CARRY_FRAMES - 1, 1)
            t_smooth = 0.5 - 0.5 * math.cos(math.pi * t)  # ease-in-out
            pos = spawn_pos + (hover_pos - spawn_pos) * t_smooth
            env._set_box_pose(placed_count, pos, quat, device)
            _step_and_capture(frames)
            frame_count += 1
        
        # =================================================================
        # Phase 3: Controlled kinematic descent (hover → release_z)
        # =================================================================
        # Lower box at constant speed, stopping ABOVE contact.
        # Each frame: move down by drop_speed_mps * frame_dt.
        # NEVER penetrate the target surface.
        descent_start_z = hover_pos[2].item()
        descent_dist = max(descent_start_z - release_z, 0.0)
        frame_dt = 1.0 / args.fps
        # Number of frames for descent at configured speed
        descent_speed_per_frame = args.drop_speed_mps * frame_dt
        descent_frames = max(1, int(descent_dist / max(descent_speed_per_frame, 1e-9)))
        
        for f in range(descent_frames):
            if frame_count >= total_frames:
                break
            frac = f / max(descent_frames - 1, 1)
            current_z = descent_start_z + (release_z - descent_start_z) * frac
            pos = target_pos.clone()
            pos[2] = current_z
            env._set_box_pose(placed_count, pos, quat, device)
            _step_and_capture(frames)
            frame_count += 1
        
        # =================================================================
        # Phase 3b: Retry demo (intentional bad placement — optional)
        # =================================================================
        if is_retry_box:
            print(f"  [RETRY] Box {box_id}: simulating failed placement + retry")
            bad_pos = target_pos.clone()
            bad_pos[0] += 0.25  # overhang
            bad_pos[2] = release_z + 0.1
            
            for f in range(RETRY_BACKUP_FRAMES):
                if frame_count >= total_frames:
                    break
                env._set_box_pose(placed_count, bad_pos, quat, device)
                _step_and_capture(frames)
                frame_count += 1
            
            # Move back to correct position above target
            for f in range(RETRY_REAPPROACH_FRAMES):
                if frame_count >= total_frames:
                    break
                t = f / max(RETRY_REAPPROACH_FRAMES - 1, 1)
                t_smooth = 0.5 - 0.5 * math.cos(math.pi * t)
                pos_interp = bad_pos.clone()
                pos_interp[0] = bad_pos[0] + (target_pos[0] - bad_pos[0]) * t_smooth
                pos_interp[1] = bad_pos[1] + (target_pos[1] - bad_pos[1]) * t_smooth
                pos_interp[2] = bad_pos[2] + (release_z - bad_pos[2]) * t_smooth
                env._set_box_pose(placed_count, pos_interp, quat, device)
                _step_and_capture(frames)
                frame_count += 1
        
        # =================================================================
        # Phase 4: RELEASE + SETTLE (pure physics — NO POSE WRITES)
        # =================================================================
        # The box is at release_z (~1cm above resting). We now let it go:
        # just step physics. The box drops the tiny gap under gravity and
        # settles naturally with the high-friction/damping mockup config.
        # CRITICAL: do NOT call _set_box_pose here. That was the root cause
        # of the snapping/down-up teleport behavior.
        for f in range(args.settle_frames):
            if frame_count >= total_frames:
                break
            _step_and_capture(frames)
            frame_count += 1
        
        # =================================================================
        # Phase 5: Pause (admire the placement)
        # =================================================================
        for f in range(PAUSE_FRAMES):
            if frame_count >= total_frames:
                break
            _step_and_capture(frames)
            frame_count += 1
        
        placed_count += 1
        print(f"  Placed box {box_id} ({placed_count}/{len(placements)}) — frame {frame_count}/{total_frames}")
    
    # Fill remaining frames with static scene
    while frame_count < total_frames:
        _step_and_capture(frames)
        frame_count += 1
    
    # =========================================================================
    # Write video
    # =========================================================================
    print(f"\n[INFO] Writing {len(frames)} frames to {args.output_path}...")
    
    try:
        import cv2
        
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        
        if len(frames) > 0:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.output_path, fourcc, args.fps, (w, h))
            
            for frame in frames:
                # Convert RGB → BGR for OpenCV
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
            
            writer.release()
            print(f"[SUCCESS] Video saved: {args.output_path} ({len(frames)} frames, {len(frames)/args.fps:.1f}s)")
        else:
            print("[WARNING] No frames captured!")
    
    except ImportError:
        print("[WARNING] cv2 not available, trying imageio...")
        try:
            import imageio
            os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
            imageio.mimwrite(args.output_path, frames, fps=args.fps)
            print(f"[SUCCESS] Video saved: {args.output_path}")
        except ImportError:
            print("[ERROR] Neither cv2 nor imageio available. Install one: pip install opencv-python")
    
    # Cleanup
    env.close()
    simulation_app.close()


# =============================================================================
# Helper: Set box pose (added to PalletTask but defined here as fallback)
# =============================================================================

# Monkey-patch PalletTask with a _set_box_pose method if not present
_original_init = PalletTask.__init__


def _patched_init(self, *a, **kw):
    _original_init(self, *a, **kw)
    
    def _set_box_pose(box_local_idx: int, pos: torch.Tensor, quat_wxyz: torch.Tensor, device: str):
        """Set pose for a single box in env 0 (mockup mode, single env).
        
        Also zeros linear and angular velocities to prevent residual momentum
        from causing jitter or unnatural motion after teleporting.
        """
        try:
            boxes = self.scene["boxes"]
            # For single env: object_pos_w is (1, max_boxes, 3)
            boxes.data.object_pos_w[0, box_local_idx] = pos.to(device)
            boxes.data.object_quat_w[0, box_local_idx] = quat_wxyz.to(device)
            # Zero velocities to prevent residual momentum
            boxes.data.object_lin_vel_w[0, box_local_idx] = 0.0
            boxes.data.object_ang_vel_w[0, box_local_idx] = 0.0
            
            # Write pose to sim
            all_pos = boxes.data.object_pos_w.reshape(-1, 3)
            all_quat = boxes.data.object_quat_w.reshape(-1, 4)
            boxes.write_object_pose_to_sim(
                torch.cat([all_pos, all_quat], dim=-1)
            )
            # Write velocities to sim
            all_lin = boxes.data.object_lin_vel_w.reshape(-1, 3)
            all_ang = boxes.data.object_ang_vel_w.reshape(-1, 3)
            boxes.write_object_velocity_to_sim(
                torch.cat([all_lin, all_ang], dim=-1)
            )
        except Exception as e:
            print(f"[WARNING] _set_box_pose failed: {e}")
    
    self._set_box_pose = _set_box_pose


PalletTask.__init__ = _patched_init


if __name__ == "__main__":
    main()
