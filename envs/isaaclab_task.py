
"""
Isaac Lab vectorized task for palletization.
This module provides:
- IsaacLabVecEnv: a vectorized environment backed by Isaac Lab.

Design notes:
- Observation is built from the internal height-map proxy channels.
- Physics is used for micro-simulation validation after placement.
- Actions: (pick, yaw, x, y).
"""
from typing import Any, Dict, Tuple, List
import numpy as np
import torch
import os

from . import pallet_task
from . import heightmap_channels

class IsaacLabVecEnv:
    def __init__(self, cfg:Dict):
        # Deferred import to avoid import errors outside Isaac
        import omni.isaac.lab as lab
        from omni.isaac.core import World
        from omni.isaac.core.utils.prims import create_prim
        from pxr import UsdGeom, Gf, UsdPhysics

        self.cfg = cfg
        self.num_envs = cfg["env"]["num_envs"]
        self.L, self.W = cfg["env"]["grid"]
        self.unit_cm = 1.0  # grid in centimeters
        self.cell_cm_x = cfg["env"]["pallet_size_cm"][0] / self.L
        self.cell_cm_y = cfg["env"]["pallet_size_cm"][1] / self.W

        # Observation channels: 8 proxy + 5 box channels
        self.C = 8 + 5
        self.obs_shape = (self.C, self.L, self.W)

        self.world = World(stage_units_in_meters=1.0)
        self.stage = self.world.stage

        # Pallet size in meters
        self.pallet_L_m = cfg["env"]["pallet_size_cm"][0] / 100.0
        self.pallet_W_m = cfg["env"]["pallet_size_cm"][1] / 100.0
        
        # Layout envs in a grid
        grid_cols = int(np.ceil(np.sqrt(self.num_envs)))
        spacing = max(self.pallet_L_m, self.pallet_W_m) * 2.0
        
        self.env_roots = []
        self.env_origins = []
        for i in range(self.num_envs):
            r = i // grid_cols
            c = i % grid_cols
            x = c * spacing
            y = r * spacing
            root_path = f"/World/env_{i:03d}"
            create_prim(root_path, "Xform")
            
            # Create pallet base
            pallet_path = f"{root_path}/pallet"
            create_prim(pallet_path, "Cube")
            UsdGeom.Cube.Get(self.stage, pallet_path).CreateSizeAttr(1.0)
            thickness = 0.15
            xf = UsdGeom.Xformable(UsdGeom.Xform.Get(self.stage, pallet_path))
            xf.AddScaleOp().Set(Gf.Vec3f(self.pallet_L_m, self.pallet_W_m, thickness))
            xf.AddTranslateOp().Set(Gf.Vec3f(x, y, thickness * 0.5))
            
            self.env_roots.append(root_path)
            self.env_origins.append((x, y))

        self.world.reset()
        self._t = 0
        
        # Internal state
        self.height_cm = np.zeros((self.num_envs, self.L, self.W), dtype=np.float32)
        self.occ = np.zeros_like(self.height_cm, dtype=np.uint8) # Changed to uint8 for consistency
        self.density_proj = np.zeros_like(self.height_cm, dtype=np.float32)
        self.stiffness_proj = np.zeros_like(self.height_cm, dtype=np.float32)
        
        self.buffer_N = cfg["env"]["buffer_N"]
        self.yaw_orients = cfg["env"]["yaw_orients"]
        self._rng = np.random.default_rng(cfg["env"]["seed"])
        
        # Track current box for each env
        self.current_boxes = [None] * self.num_envs # Dict with box params
        self.current_box_prims = [None] * self.num_envs # Prim path

    def _spawn_next_box(self, env_id:int):
        """Selects and spawns the next box for the environment."""
        # Random box dimensions (simplified for now)
        # In a real scenario, this would come from a dataset or distribution
        L_box = self._rng.uniform(0.2, 0.5)
        W_box = self._rng.uniform(0.2, 0.5)
        H_box = self._rng.uniform(0.1, 0.3)
        
        box_params = {"L": L_box, "W": W_box, "H": H_box, "id": self._rng.integers(0, 100000)}
        self.current_boxes[env_id] = box_params
        
        # Spawn prim (hidden or at spawn location)
        from omni.isaac.core.utils.prims import create_prim
        from pxr import UsdGeom, Gf
        
        root = self.env_roots[env_id]
        origin = self.env_origins[env_id]
        prim_path = f"{root}/box_{self._t}_{env_id}" # Unique path per step
        create_prim(prim_path, "Cube")
        
        UsdGeom.Cube.Get(self.stage, prim_path).CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(UsdGeom.Xform.Get(self.stage, prim_path))
        xf.AddScaleOp().Set(Gf.Vec3f(L_box, W_box, H_box))
        
        # Place high above or aside until action
        xf.AddTranslateOp().Set(Gf.Vec3f(origin[0], origin[1], 5.0)) 
        
        self.current_box_prims[env_id] = prim_path
        return box_params

    def reset(self):
        self.world.reset()
        self._t = 0
        self.height_cm.fill(0.0)
        self.occ.fill(0)
        self.density_proj.fill(0.0)
        self.stiffness_proj.fill(0.0)
        
        for i in range(self.num_envs):
            self._spawn_next_box(i)
            
        return self._build_obs()

    def _build_obs(self):
        # Compute channels for all envs
        obs_list = []
        for i in range(self.num_envs):
            box = self.current_boxes[i]
            # Convert box to cells
            box_l_cells = box["L"] * 100.0 / self.cell_cm_x
            box_w_cells = box["W"] * 100.0 / self.cell_cm_y
            
            channels = heightmap_channels.compute_channels(
                self.height_cm[i], self.occ[i], 
                box_l_cells, box_w_cells, 
                self.density_proj[i], self.stiffness_proj[i]
            )
            
            # Stack channels: H, Occ, Support, Roughness, Density, Stiffness, Safety, Cap (8 channels)
            # Order must match model expectation. 
            # Model expects 8 + 5.
            # Let's assume the order: H, Occ, Support, Roughness, Density, Stiffness, Safety, HeightCap
            c_stack = np.stack([
                channels["H"], channels["Occ"], channels["Support"], channels["Roughness"],
                channels["Density_proj"], channels["Stiffness_proj"], channels["SafetyMargin"], channels["HeightCap"]
            ])
            
            # Add box info channels (5 channels: L, W, H, Density, Stiffness) - broadcasted
            # Normalized box features
            box_feat = np.array([box["L"], box["W"], box["H"], 1.0, 1.0], dtype=np.float32).reshape(5, 1, 1)
            box_channels = np.tile(box_feat, (1, self.L, self.W))
            
            full_obs = np.concatenate([c_stack, box_channels], axis=0)
            obs_list.append(full_obs)
            
        return np.stack(obs_list)

    def step(self, actions):
        """
        actions: (N, 4) numpy array -> [pick_id, yaw_id, x_idx, y_idx]
        """
        from pxr import UsdGeom, Gf
        
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]
        
        for i in range(self.num_envs):
            action = actions[i]
            # pick_id = action[0] # Not used yet (single box buffer)
            yaw_id = int(action[1])
            x_idx = int(action[2])
            y_idx = int(action[3])
            
            # 1. Move box to target
            box = self.current_boxes[i]
            prim_path = self.current_box_prims[i]
            origin = self.env_origins[i]
            
            # Grid to meters
            # Center of the cell
            pos_x = origin[0] + (x_idx + 0.5) * (self.cell_cm_x / 100.0) - (self.pallet_L_m / 2.0) + (self.pallet_L_m/2.0) # Wait, origin is center? No, origin is corner?
            # Let's assume origin is center of pallet for simplicity in Xform, but grid is 0..L
            # In __init__, we set translate to (x, y, thickness*0.5). 
            # If pallet is scaled by L, W, it is centered at (0,0,0) locally?
            # UsdGeom.Cube is -1..1 or 0..1? Default is -1..1 (size 2). But we used CreateSizeAttr(1.0).
            # So it's -0.5..0.5.
            # So pallet extends from x-L/2 to x+L/2.
            # Grid 0 is at x - L/2.
            
            # Correct logic:
            pallet_min_x = origin[0] - self.pallet_L_m / 2.0
            pallet_min_y = origin[1] - self.pallet_W_m / 2.0
            
            target_x = pallet_min_x + (x_idx + 0.5) * (self.cell_cm_x / 100.0)
            target_y = pallet_min_y + (y_idx + 0.5) * (self.cell_cm_y / 100.0)
            
            # Get current max height at that location to stack on top
            # Simplified: just drop from high up
            drop_z = 2.0 
            
            # Apply Pose
            xf = UsdGeom.Xformable(UsdGeom.Xform.Get(self.stage, prim_path))
            # Reset transforms and apply new ones
            xf.ClearXformOpOrder()
            xf.AddScaleOp().Set(Gf.Vec3f(box["L"], box["W"], box["H"]))
            
            # Yaw
            yaw_deg = self.yaw_orients[yaw_id] if yaw_id < len(self.yaw_orients) else 0.0
            xf.AddRotateZOp().Set(yaw_deg)
            
            xf.AddTranslateOp().Set(Gf.Vec3f(target_x, target_y, drop_z))
            
        # 2. Step Physics (Micro-sim)
        # Run for a few steps to let boxes settle
        for _ in range(10): 
            self.world.step(render=False)
            
        # 3. Compute Reward & Update State
        for i in range(self.num_envs):
            # Analyze stability/placement
            # For now, using dummy MicroSimResult
            # In real impl, check physics state (velocity, contact)
            
            # Update Heightmap (Simplified: just add box height to grid)
            # This is a heuristic update. Real update should raycast or query physics.
            box = self.current_boxes[i]
            x_idx = int(actions[i, 2])
            y_idx = int(actions[i, 3])
            
            # Update internal heightmap
            l_cells = int(box["L"] * 100.0 / self.cell_cm_x)
            w_cells = int(box["W"] * 100.0 / self.cell_cm_y)
            
            # Handle Yaw (swap L/W if 90 deg)
            yaw_id = int(actions[i, 1])
            if yaw_id == 1: # 90 deg
                l_cells, w_cells = w_cells, l_cells
                
            # Naive height update
            h_add = box["H"] * 100.0 # cm
            
            # Clip to bounds
            x_start = max(0, x_idx - l_cells//2)
            x_end = min(self.L, x_idx + l_cells//2)
            y_start = max(0, y_idx - w_cells//2)
            y_end = min(self.W, y_idx + w_cells//2)
            
            if x_end > x_start and y_end > y_start:
                current_max = np.max(self.height_cm[i, x_start:x_end, y_start:y_end])
                new_h = current_max + h_add
                self.height_cm[i, x_start:x_end, y_start:y_end] = new_h
                self.occ[i, x_start:x_end, y_start:y_end] = 1
            
            # Compute Reward
            ms = pallet_task.MicroSimResult(True, 0.0, 0.0, 0.0) # Assume stable
            r = pallet_task.compute_reward(self.cfg, 0.1, ms, False)
            rewards[i] = r
            
            # Compute Heuristics
            heuristics = pallet_task.compute_heuristics(
                self.height_cm[i], 
                self.L * self.W, 
                self.cfg["env"]["pallet_size_cm"][2]
            )
            infos[i].update(heuristics)
            
            # Spawn next box
            self._spawn_next_box(i)
            
        self._t += 1
        obs = self._build_obs()
        
        return obs, rewards, dones, infos
