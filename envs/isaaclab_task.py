"""
Isaac Lab vectorized task for palletization.
This module provides:
- IsaacLabVecEnv: a vectorized environment backed by Isaac Lab (PhysX).

Refactored for High-Fidelity Physics:
- True physics settling loop.
- Raycast-based heightmap perception.
- Domain randomization (mass, friction, restitution).
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
        from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema
        from omni.physx import get_physx_scene_query_interface
        
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
        
        self.physx_query_interface = get_physx_scene_query_interface()

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
            
            # Add Static Physics to pallet
            UsdPhysics.CollisionAPI.Apply(self.stage.GetPrimAtPath(pallet_path))
            # Make it a static rigid body? Or just collision for static? 
            # Usually we need RigidBodyAPI for dynamics, but for static ground just Collision is enough, 
            # assuming implicit static body.
            
            self.env_roots.append(root_path)
            self.env_origins.append((x, y))

        self.world.reset()
        self._t = 0
        
        # Internal state
        self.height_cm = np.zeros((self.num_envs, self.L, self.W), dtype=np.float32)
        self.occ = np.zeros_like(self.height_cm, dtype=np.uint8)
        self.density_proj = np.zeros_like(self.height_cm, dtype=np.float32)
        self.stiffness_proj = np.zeros_like(self.height_cm, dtype=np.float32)
        
        self.buffer_N = cfg["env"]["buffer_N"]
        self.yaw_orients = cfg["env"]["yaw_orients"]
        self._rng = np.random.default_rng(cfg["env"]["seed"])
        
        # Track current box for each env
        self.current_boxes = [None] * self.num_envs # Dict with box params
        self.current_box_prims = [None] * self.num_envs # Prim path

    def _spawn_next_box(self, env_id:int):
        """Selects and spawns the next box for the environment with Domain Randomization."""
        from omni.isaac.core.utils.prims import create_prim
        from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema
        
        # Random box dimensions
        L_box = self._rng.uniform(0.2, 0.5)
        W_box = self._rng.uniform(0.2, 0.5)
        H_box = self._rng.uniform(0.1, 0.3)
        
        # Domain Randomization: Physics Properties
        mass = self._rng.uniform(1.0, 10.0)
        dynamic_friction = self._rng.uniform(0.3, 0.9)
        restitution = self._rng.uniform(0.0, 0.4)
        
        box_params = {
            "L": L_box, "W": W_box, "H": H_box, 
            "id": self._rng.integers(0, 100000),
            "mass": mass, "friction": dynamic_friction, "restitution": restitution
        }
        self.current_boxes[env_id] = box_params
        
        root = self.env_roots[env_id]
        origin = self.env_origins[env_id]
        prim_path = f"{root}/box_{self._t}_{env_id}" 
        create_prim(prim_path, "Cube")
        
        prim = self.stage.GetPrimAtPath(prim_path)
        UsdGeom.Cube.Get(self.stage, prim_path).CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(UsdGeom.Xform.Get(self.stage, prim_path))
        xf.AddScaleOp().Set(Gf.Vec3f(L_box, W_box, H_box))
        
        # Physics APIs
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        massAPI = UsdPhysics.MassAPI.Apply(prim)
        massAPI.CreateMassAttr(mass)
        
        # Material
        mat_path = f"{prim_path}/Material"
        create_prim(mat_path, "Material")
        # In USD Physics, we define material and bind it.
        # Or simpler: use PhysicsMaterialAPI on the prim if supported, but typically we create a Material prim with PhysicsMaterialAPI.
        
        mat_prim = self.stage.GetPrimAtPath(mat_path)
        PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
        mat_api = UsdPhysics.MaterialAPI.Apply(mat_prim)
        mat_api.CreateDynamicFrictionAttr(dynamic_friction)
        mat_api.CreateStaticFrictionAttr(dynamic_friction) # assume similar
        mat_api.CreateRestitutionAttr(restitution)
        
        # Bind material
        rel = UsdPhysics.MaterialBindingAPI.Apply(prim)
        rel.Bind(mat_prim, purpose="physics")

        # Place high above or aside
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
            
            c_stack = np.stack([
                channels["H"], channels["Occ"], channels["Support"], channels["Roughness"],
                channels["Density_proj"], channels["Stiffness_proj"], channels["SafetyMargin"], channels["HeightCap"]
            ])
            
            # Add box info channels
            box_feat = np.array([box["L"], box["W"], box["H"], box.get("mass", 1.0)/10.0, box.get("friction", 0.5)], dtype=np.float32).reshape(5, 1, 1)
            box_channels = np.tile(box_feat, (1, self.L, self.W))
            
            full_obs = np.concatenate([c_stack, box_channels], axis=0)
            obs_list.append(full_obs)
            
        return np.stack(obs_list)

    def _update_heightmap_physx(self):
        """Raycast based heightmap update using PhysX."""
        from pxr import Gf
        
        # Batching queries might be complex with Python API loop, so we iterate envs.
        # Ideally we use vectorized raycasts if available, but here we do per-env grid loop or per-env sparse raycast?
        # A full grid raycast (L x W) * num_envs is expensive.
        # But for correctness we need it. 
        # Optimized: Only raycast near recent action? No, boxes might topple anywhere.
        
        for i in range(self.num_envs):
            origin = self.env_origins[i]
            # Raycast grid
            # Center of each cell
            # We can use a simpler approach: raycast from top down for each cell.
            
            for r in range(self.L):
                for c in range(self.W):
                    # World coords
                    x = origin[0] - self.pallet_L_m/2.0 + (r + 0.5) * (self.cell_cm_x/100.0)
                    y = origin[1] - self.pallet_W_m/2.0 + (c + 0.5) * (self.cell_cm_y/100.0)
                    rotation_y = 0.0 # Just vertical rays
                    
                    ray_origin = Gf.Vec3d(x, y, 3.0) # Start from 3m
                    ray_dir = Gf.Vec3d(0, 0, -1)
                    
                    # Raycast
                    hit = self.physx_query_interface.raycast_closest(ray_origin, ray_dir, 10.0)
                    
                    if hit["hit"]:
                        dist = hit["distance"]
                        # z of hit
                        z_hit = 3.0 - dist
                        self.height_cm[i, r, c] = max(0.0, z_hit * 100.0)
                        if z_hit > 0.05: # above pallet thickness
                             self.occ[i, r, c] = 1
                        else:
                             self.occ[i, r, c] = 0
                    else:
                        self.height_cm[i, r, c] = 0.0
                        self.occ[i, r, c] = 0

    def step(self, actions):
        from pxr import UsdGeom, Gf
        from omni.isaac.core.utils.prims import get_prim_pose
        
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]
        
        for i in range(self.num_envs):
            action = actions[i]
            yaw_id = int(action[1])
            x_idx = int(action[2])
            y_idx = int(action[3])
            
            box = self.current_boxes[i]
            prim_path = self.current_box_prims[i]
            origin = self.env_origins[i]
            
            pallet_min_x = origin[0] - self.pallet_L_m / 2.0
            pallet_min_y = origin[1] - self.pallet_W_m / 2.0
            
            target_x = pallet_min_x + (x_idx + 0.5) * (self.cell_cm_x / 100.0)
            target_y = pallet_min_y + (y_idx + 0.5) * (self.cell_cm_y / 100.0)
            
            # Simple heuristic for drop height: max height in target area + margin
            l_cells = int(box["L"] * 100.0 / self.cell_cm_x)
            w_cells = int(box["W"] * 100.0 / self.cell_cm_y)
            if yaw_id == 1:
                l_cells, w_cells = w_cells, l_cells
                
            x_start = max(0, x_idx - l_cells//2)
            x_end = min(self.L, x_idx + l_cells//2)
            y_start = max(0, y_idx - w_cells//2)
            y_end = min(self.W, y_idx + w_cells//2)
            
            if x_end > x_start and y_end > y_start:
                current_max_h = np.max(self.height_cm[i, x_start:x_end, y_start:y_end])
            else:
                current_max_h = 0.0
                
            drop_z = (current_max_h / 100.0) + box["H"]/2.0 + 0.05 # small margin
            drop_z = max(drop_z, box["H"]/2.0 + 0.15) # At least on pallet + margin
            
            # Apply Initial Pose for Drop
            xf = UsdGeom.Xformable(UsdGeom.Xform.Get(self.stage, prim_path))
            xf.ClearXformOpOrder()
            xf.AddScaleOp().Set(Gf.Vec3f(box["L"], box["W"], box["H"]))
            
            yaw_deg = self.yaw_orients[yaw_id] if yaw_id < len(self.yaw_orients) else 0.0
            xf.AddRotateZOp().Set(yaw_deg)
            xf.AddTranslateOp().Set(Gf.Vec3f(target_x, target_y, drop_z))

            # Store target z for stability check
            self.current_boxes[i]["target_z"] = drop_z
            
        # 2. Physics Settling Loop
        for _ in range(20): 
            self.world.step(render=False)

        # 3. Update Perception (Raycast)
        self._update_heightmap_physx()
            
        # 4. Compute Reward & Verify Stability
        for i in range(self.num_envs):
            box = self.current_boxes[i]
            prim_path = self.current_box_prims[i]
            
            # Check actual pose
            actual_pos, actual_quat = get_prim_pose(prim_path) # numpy arrays
            
            # Stability Check
            # 1. Height check: if it fell significantly below target drop, it might have toppled or slid off high stack
            # Tolerance: maybe 5cm?
            z_diff = input_target_z = box["target_z"] - actual_pos[2]
            
            # 2. Rotation check
            # Convert quat to euler (or check up vector alignment)
            # Quat is (w, x, y, z)
            w, x, y, z = actual_quat
            # rot matrix Z axis (up vector of box)
            # R_22 = 1 - 2(x^2 + y^2)
            # If box creates angle with vertical > 15 deg
            # Cos(theta) = z_axis . vertical(0,0,1)
            # Box Z axis local is (0,0,1). In world:
            # z_world = 2(xz + wy) ?? No
            # R = ...
            # 3rd column of R:
            # 2(xz + wy), 2(yz - wx), 1 - 2(x^2 + y^2)
            # We want the 3rd component (dot product with 0,0,1)
            vertical_dot = 1.0 - 2.0 * (x**2 + y**2)
            
            is_tilted = vertical_dot < np.cos(np.deg2rad(15.0))
            has_fallen = z_diff > 0.10 # 10cm drop from expected = unstable/fell into gap
            
            stable = not (is_tilted or has_fallen)
            
            ms = pallet_task.MicroSimResult(stable, 0.0, 0.0, 0.0) 
            
            # Reward
            r = pallet_task.compute_reward(self.cfg, 0.1, ms, False) # simplified vol frac
            rewards[i] = r
            
            heuristics = pallet_task.compute_heuristics(
                self.height_cm[i], 
                self.L * self.W, 
                self.cfg["env"]["pallet_size_cm"][2]
            )
            infos[i].update(heuristics)
            infos[i]["stable"] = stable
            
            self._spawn_next_box(i)

        self._t += 1
        obs = self._build_obs()
        
        return obs, rewards, dones, infos
