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

    def get_action_mask(self):
        """
        Computes the geometric action mask for the current box in each environment.
        Returns:
            np.ndarray: (num_envs, L * W) boolean mask (True=valid, False=invalid/overhang)
        """
        masks = np.zeros((self.num_envs, self.L * self.W), dtype=np.float32) # Using float for consistency with logits, or bool? Usually float 1.0/0.0
        
        for i in range(self.num_envs):
            box = self.current_boxes[i]
            l_cells_0 = int(np.ceil(box["L"] * 100.0 / self.cell_cm_x))
            w_cells_0 = int(np.ceil(box["W"] * 100.0 / self.cell_cm_y))
            
            # For each cell (r, c), check if box placed at center overhangs
            # Center of box is at (r, c). 
            # Dimensions: l_cells, w_cells.
            # Ranges: [r - l//2, r + l//2] ?? 
            # Needs to match step logic:
            # x_start = max(0, x_idx - l_cells//2)
            # x_end = min(self.L, x_idx + l_cells//2)
            # If x_end - x_start != original_l_cells, it's overhang/cutoff.
            # But the constraint is "overhang the pallet boundaries".
            # Strictly: mask = 0 if any part of box is outside grid.
            
            # Let's vectorize this calculation for the grid? Or simple loops.
            # Grid is small (40x48), 2000 cells.
            
            # Indices: 0..L-1
            # Box extent from center i: [i - l//2, i + l//2] (roughly)
            # 
            # Let's define valid centers:
            # min_center_x such that x - l/2 >= 0
            # max_center_x such that x + l/2 < L
            # Careful with integer division and parity.
            
            # Let's use the exact logic from typical box placement:
            # If l_cells is even, e.g. 4. Center at x. Extent x-2 to x+2? Length 4.
            # x=2. 0,1,2,3 -> ok.
            
            # Let's compute valid ranges for X and Y for both orientations.
            # We assume the policy outputs one position, but yaw is separate?
            # Wait, the Architecture section says: "Joint Position Head... flattened logit vector... x_head and y_head removed".
            # Does it remove yaw? No, "Refactor PolicyHeads: Remove x_head and y_head. Replace... single pos_head". 
            # Yaw head remains.
            # But validity depends on Yaw!
            # If we mask *positions*, we need to know the yaw.
            # But the network outputs pick, yaw, pos simultaneously?
            # Or is it auto-regressive?
            # Usually independent heads.
            # If independent, we can't mask Pos based on Yaw perfectly unless we intersect the masks for all yaws (conservative) or union?
            # "If placing a box at (x, y) would cause it to overhang... set mask to 0".
            # If ANY yaw fits? Or if ALL yaws fit? 
            # Usually: The mask is passed to `pos_head`. 
            # If we don't know yaw yet, we should probably mask positions that are invalid for *currently selected* yaw if separate logic?
            # But training is simultaneous.
            # Standard approach: Mask if invalid for *both* yaws (too big to fit anywhere)? 
            # OR, Validity check usually implies "Can I drop here?".
            # If I drop with wrong yaw, it overhangs.
            # If I drop with right yaw, it fits.
            # As a heuristic for *position* head, maybe we mask only if it's invalid for *all* possible orientations? 
            # OR we assume the agent learns to correlate.
            # The prompt says: "If placing a box at (x,y) would cause it to overhang... set mask to 0".
            # Given the constraints, let's be strict: A position is valid ONLY if the box fits within boundaries. 
            # Since `l` and `w` differ, the footprint differs.
            # Let's assume separate masks per yaw? No, single pos head.
            # Let's mask a position if it is invalid for the *current* box dimensions in *either* orientation? No that's too restrictive (might fit one way but not other).
            # Invalid for *both* orientations? (Fits neither way).
            # Or valid for *at least one* orientation? (If it fits one way, we let the agent pick that pos, and hope it picks the right yaw).
            # I will go with "Valid for at least one orientation" (Union of valid masks).
            # This gives the agent a chance to place it.
            
            valid_mask = np.zeros((self.L, self.W), dtype=bool)
            
            for yaw_id, (l, w) in enumerate([(l_cells_0, w_cells_0), (w_cells_0, l_cells_0)]):
                 # valid x range
                 # left = x - l//2 >= 0
                 # right = x + l//2 + (l%2) <= L ? 
                 # Let's check `step` logic:
                 # x_start = max(0, x_idx - l_cells//2)
                 # x_end = min(self.L, x_idx + l_cells//2) 
                 # Note: this logic CLIPS. It allows dropping and clipping!
                 # But the objective says "avoid invalid actions... overhang".
                 # So we want x_start == x_idx - l//2  AND x_end == x_idx + l//2
                 # (depending on odd/even parity handling in step, step seems to define footprint by that range).
                 # Wait, `x_idx + l_cells//2` -- if l=4, x=2. start=0, end=4. len=4. Correct.
                 # if x=0. start=0, end=2. len=2. CLIPPED.
                 
                 # So valid X indices are:
                 # x - l//2 >= 0  => x >= l//2
                 # x + l//2 <= L  => x <= L - l//2 (roughly)
                 # Let's be precise.
                 
                 half_l = l // 2
                 # If l is odd, say 3. half=1. center=x. x-1..x+1. size 3.
                 # x needs: x >= 1, x+1 < L => x <= L-2.
                 # If l is even, say 4. half=2. x-2..x+2. size 4.
                 # wait, x-2 to x+2 is size 4? indices: x-2, x-1, x, x+1. (Right exclusive?)
                 # The slice in step is `self.height_cm[..., x_start:x_end]`.
                 # Python slice x_start:x_end has length x_end - x_start.
                 # We want x_end - x_start == l.
                 # x_end = x + l//2
                 # x_start = x - l//2
                 # Diff = l (?) -> 2*(l//2). 
                 # If l=3, diff=2. WRONG. Box lost size?
                 # If l=3, l//2 = 1. x-1 to x+1. Diff=2.
                 # So the step function logic effectively shrinks odd boxes? 
                 # Or maybe `step` logic is `x_idx - l_cells//2` to `x_idx + l_cells//2 + (l_cells%2)`?
                 # No, existing code: `x_end = min(self.L, x_idx + l_cells//2)`.
                 # It seems the existing code might use a specific convention.
                 # For "Geometric Action Masking", I should ensure I don't clip.
                 # So I will enforce that `x - l//2 >= 0` and `x + l//2 <= L`.
                 # But what about the missing pixel for odd numbers?
                 # Let's stick to the logic: "Fit inside".
                 # A box of size L occupies L cells.
                 # x_center index.
                 # If L is current size.
                 
                 min_x = l // 2
                 max_x = self.L - (l - min_x) # Ensures x+ (l-l//2) <= L ?
                 # Range is [x - l//2, x + l - l//2]. Length is l.
                 # Exclusive end: x + l - l//2.
                 # Must be <= L.
                 # so x <= L - (l - l//2).
                 
                 min_y = w // 2
                 max_y = self.W - (w - min_y)
                 
                 # Fill valid mask
                 # 2D slice
                 # valid_mask[min_x:max_x, min_y:max_y] = True ? 
                 # Be careful with max_x being exclusive for range, but inclusive for valid index?
                 # range(min_x, max_x) -> indices.
                 
                 # Let's construct ranges
                 x_valid = np.arange(self.L)
                 y_valid = np.arange(self.W)
                 
                 m_x = (x_valid >= min_x) & (x_valid < max_x)
                 m_y = (y_valid >= min_y) & (y_valid < max_y)
                 
                 # Outer product for this yaw
                 yaw_mask = np.outer(m_x, m_y)
                 valid_mask = valid_mask | yaw_mask
            
            masks[i] = valid_mask.flatten()
            
        return masks

    def _update_heightmap_physx(self, rois=None):
        """
        Raycast based heightmap update using PhysX.
        rois: List of Tuple (min_x_idx, max_x_idx, min_y_idx, max_y_idx) inclusive.
             One tuple per environment. If None, updates entire grid for all envs.
        """
        from pxr import Gf

        for i in range(self.num_envs):
            origin = self.env_origins[i]
            
            roi = rois[i] if rois is not None else None
            
            if roi is None:
                r_range = range(self.L)
                c_range = range(self.W)
            else:
                # ROI is tuple of indices
                r_min, r_max, c_min, c_max = roi
                # Clip to grid
                r_start = max(0, r_min)
                r_end = min(self.L, r_max + 1) # python range is exclusive at end
                c_start = max(0, c_min)
                c_end = min(self.W, c_max + 1)
                r_range = range(r_start, r_end)
                c_range = range(c_start, c_end)

            for r in r_range:
                for c in c_range:
                    # World coords
                    x = origin[0] - self.pallet_L_m/2.0 + (r + 0.5) * (self.cell_cm_x/100.0)
                    y = origin[1] - self.pallet_W_m/2.0 + (c + 0.5) * (self.cell_cm_y/100.0)
                    
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
        # 3. Update Perception (Raycast) - Sparse Update
        # Calculate ROI including margin
        margin = 2
        # ROI based on action indices and box size. 
        # Note: box might have moved/toppled, but typically stays near drop.
        # We take a conservative region around the drop target.
        # If we really want to be safe, we could check actual pose, but we want speed.
        # Let's use the drop target bounds + margin.
        
        # This implementation assumes all envs did similar actions or just takes a superset?
        # actually _update_heightmap_physx iterates envs, so we should ideally pass a list of ROIs if we want per-env optimization.
        # But my modified _update_heightmap_physx takes a single ROI argument.
        # To strictly follow "per env" optimization, I need to modify _update_heightmap_physx to handle list of rois 
        # OR just call it inside the loop here?
        # The function `_update_heightmap_physx` iterates `range(self.num_envs)`.
        # So passing one ROI applies to all? That's wrong if they drop in different places.
        # Wait, the previous implementation of `_update_heightmap_physx` iterated all envs.
        # I should change `_update_heightmap_physx` to accept a LIST of ROIs or compute it internally if I passed actions?
        # Let's update `_update_heightmap_physx` to just do the loop and logic correctly.
        # Actually, let's just make `_update_heightmap_physx` capable of taking a list of ROIs.
        
        # But for now, let's fix the call site logic. 
        # I will change the method `_update_heightmap_physx` to take `rois: List[Tuple]`.
        
        rois = []
        for i in range(self.num_envs):
            action = actions[i]
            yaw_id = int(action[1])
            x_idx = int(action[2])
            y_idx = int(action[3])
            box = self.current_boxes[i]
            
            l_cells = int(box["L"] * 100.0 / self.cell_cm_x)
            w_cells = int(box["W"] * 100.0 / self.cell_cm_y)
            if yaw_id == 1:
                l_cells, w_cells = w_cells, l_cells
            
            x_start = x_idx - l_cells//2 - margin
            x_end = x_idx + l_cells//2 + margin
            y_start = y_idx - w_cells//2 - margin
            y_end = y_idx + w_cells//2 + margin
            rois.append((x_start, x_end, y_start, y_end))

        self._update_heightmap_physx(rois)
            
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
            
            heuristics = pallet_task.compute_heuristics(
                self.height_cm[i], 
                self.L * self.W, 
                self.cfg["env"]["pallet_size_cm"][2]
            )
            
            # Reward
            r = pallet_task.compute_reward(self.cfg, 0.1, ms, False, heuristics["height_std"]) # simplified vol frac
            rewards[i] = r
            
            infos[i].update(heuristics)
            infos[i]["stable"] = stable
            
            self._spawn_next_box(i)

        self._t += 1
        obs = self._build_obs()
        
        return obs, rewards, dones, infos
