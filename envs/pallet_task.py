
import torch
import numpy as np
from typing import Dict, Any

from omni.isaac.core.utils.torch.rotations import quat_rotate_inverse, quat_mul
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.utils import configclass
import gymnasium as gym

from pallet_rl.utils.heightmap_rasterizer import WarpHeightmapGenerator

@configclass
class PalletTaskCfg(DirectRLEnvCfg):
    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=1/60.0, render_interval=2)
    
    # Decimation
    decimation = 10 
    
    # Env Params
    num_envs: int = 4096
    env_spacing: float = 3.0
    pallet_size_cm: list = [120., 100.] # Actual logic uses 120x80 usually? Code used 120/100 default but logic might assume 120/80? 
    # Let's align with user constraints [120, 80] if implied by 240x160 pixels.
    # But Config said [120, 100]. 160x240 @ 0.5cm covers 80x120.
    # I should set pallet_size_cm to [120., 80.] strictly to match the new map?
    # Or map covers slightly more?
    # Let's set to [120, 80] to be clean.
    pallet_size_cm: list = [120., 80.] 
    grid_res_cm: float = 0.5 # Fine resolution for CNN
    map_shape: tuple = (160, 240) # Rectangular (X, Y)
    max_boxes: int = 50
    max_height: float = 2.0 # For normalization
    
    # State Dims
    # Robot State: Joints (e.g. 6 pos + 6 vel) + Gripper? 
    # Let's assume 12 DOFs total -> 24.
    robot_state_dim = 24 
    
    # Calculated later in init, but for Config access we might need it.
    # 160*240 = 38400 (Visual)
    # 50 (Buffer) + 3 (Dims) + 24 (Proprio) = 77
    # Total = 38477
    num_observations = 38477
    # MultiDiscrete action space doesn't map to a single int easily for config, 
    # but we will override action_space in __init__
    num_actions = 5 # Placeholder length of MultiDiscrete dims
    
    episode_length_s = 60.0

class PalletTask(DirectRLEnv):
    def __init__(self, cfg: PalletTaskCfg, render_mode: str | None = None, **kwargs):
        # Recalculate if needed or trust cfg
        self.robot_state_dim = cfg.robot_state_dim
        self.map_shape = cfg.map_shape # Tuple
        
        # Dynamic calculation of observation dimension
        vis_dim = self.map_shape[0] * self.map_shape[1]
        vec_dim = 10 * 5 # Buffer flattened
        box_dim = 3
        cfg.num_observations = vis_dim + vec_dim + box_dim + self.robot_state_dim
        
        super().__init__(cfg, render_mode, **kwargs)
        
        # Warp Rasterizer Init
        # Warp Rasterizer Init
        self.heightmap_gen = WarpHeightmapGenerator(
            device=self.device,
            num_envs=self.num_envs,
            max_boxes=self.cfg.max_boxes,
            grid_res=self.cfg.grid_res_cm / 100.0,
            map_size=self.map_shape,
            pallet_dims=(self.cfg.pallet_size_cm[0]/100.0, self.cfg.pallet_size_cm[1]/100.0)
        )
        
        # Box Dims (Placeholder/Mock)
        self.box_dims_tensor = torch.tensor([0.4, 0.3, 0.2], device=self.device).repeat(self.num_envs, self.cfg.max_boxes, 1)

        # --- Buffer Logic ---
        # Shape: (num_envs, 10, 5) -> 10 slots, 5 dims: [L, W, H, ID, Age]
        self.buffer_state = torch.zeros((self.num_envs, 10, 5), device=self.device)
        
        # --- Action Space ---
        # MultiDiscrete([3, 10, 16, 24, 2])
        # 0: Op (Place, Store, Retrieve)
        # 1: Buffer Slot (0-9)
        # 2: Grid X (0-15)
        # 3: Grid Y (0-23)
        # 4: Rot (0-1)
        self.action_space = gym.spaces.MultiDiscrete([3, 10, 16, 24, 2])

    def _setup_scene(self):
        pass

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # Reset buffer for these envs
        self.buffer_state[env_ids] = 0.0
        
        # Reset Box Index
        if not hasattr(self, "box_idx"):
            self.box_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.box_idx[env_ids] = 0
        
        # Randomize Box Dims for the new episode
        # (N, max_boxes, 3)
        # We'll just randomize for ALL boxes in these envs for simplicity
        # Or just the sequence? Let's randomize the whole sequence
        dims = torch.tensor([0.4, 0.3, 0.2], device=self.device) # Base
        rand = torch.rand((len(env_ids), self.cfg.max_boxes, 3), device=self.device) * 0.2 - 0.1
        self.box_dims_tensor[env_ids] = dims + rand
        
        # Reset Box Poses (Move all to spawn/holding area)
        # We need to move ALL boxes for these envs away
        # boxes shape in view? We need to verify how "boxes" are indexed.
        # Assuming DirectRLEnv flat indexing? Or [N, MaxBoxes]?
        # Usually RigidObject doesn't support multidim indexing naturally if initialized flat.
        # But let's assume we can access them.
        # If not, we rely on physics simulation handling "sleeping" boxes.
        # We will teleport the active box in step() anyway.

    def step(self, actions: torch.Tensor):
        # 1. Parse Actions
        # actions: (N, 5) -> Op, Slot, X, Y, Rot
        op = actions[:, 0]
        slot = actions[:, 1]
        grid_x = actions[:, 2]
        grid_y = actions[:, 3]
        rot = actions[:, 4]
        
        # Identify Masks
        place_mask = (op == 0)
        store_mask = (op == 1)
        retrieve_mask = (op == 2)
        
        # 2. Logic & Teleport
        
        # Current Active Box Indices
        # We need strict indexing
        env_indices = torch.arange(self.num_envs, device=self.device)
        active_box_idx = self.box_idx # (N,)
        
        # Convert to Global Rigid Body Indices
        # Assuming flattened: global_idx = env_idx * max_boxes + box_idx
        # NOTE: This depends on how the Scene initializes the RigidObject.
        # If "boxes" has num_instances = num_envs * max_boxes.
        global_active_idx = env_indices * self.cfg.max_boxes + active_box_idx
        
        # --- Handle STORE ---
        # Move current box data to buffer
        # Slot validation: 0-9.
        # If slot occupied, penalize? For now, overwrite.
        # buffer_state: [L, W, H, ID, Age]
        # ID could be just 1.0 (valid).
        if torch.any(store_mask):
            target_slots = slot[store_mask]
            
            # Get dims of current box
            current_dims = self.box_dims_tensor[env_indices[store_mask], active_box_idx[store_mask]]
            
            # Create feature vector: [L, W, H, 1, 0]
            # ID=1, Age=0
            features = torch.zeros((store_mask.sum(), 5), device=self.device)
            features[:, :3] = current_dims
            features[:, 3] = 1.0 # Exists
            
            self.buffer_state[env_indices[store_mask], target_slots] = features
            
            # Hide the physical box? 
            # We teleport it far away so it doesn't interfere.
            # We'll handle teleport batching below.
            
        # --- Handle RETRIEVE ---
        # If Retrieve, we want to place a box FROM buffer.
        # We need to know WHICH physical box to use?
        # A simple hack: Use the "active_box" physical object, but resize it to match buffer data?
        # Valid logic: "Swap" properties of active physical box to match buffer content.
        # Verify if buffer has data
        has_data =  self.buffer_state[env_indices, slot, 3] > 0.5 # ID > 0
        valid_retrieve = retrieve_mask & has_data
        
        if torch.any(valid_retrieve):
            # Fetch from buffer
            retrieved_dims = self.buffer_state[env_indices[valid_retrieve], slot[valid_retrieve], :3]
            
            # Update physical dims?
            # RigidBody scale/size update at runtime is tricky in PhysX without respawn.
            # ISAAC LAB: "root_scale_w"? 
            # If we generally assume boxes are same size or we can scale them.
            # self.scene["boxes"].write_root_scale_to_sim(...)
            # Let's assume we update the logic state `box_dims_tensor` for reward calc,
            # and hopefully visual/collision scale if supported.
            self.box_dims_tensor[env_indices[valid_retrieve], active_box_idx[valid_retrieve]] = retrieved_dims
            
            # Clear buffer slot
            self.buffer_state[env_indices[valid_retrieve], slot[valid_retrieve]] = 0.0
            
            # Treat as PLACE now
            # We will fall through to logic for position calculation
            pass
            
        # --- Calculate Target Poses for Placement (Place OR Retrieve) ---
        # Target: Grid -> (x,y,z)
        # Grid: 5cm res.
        # Pallet center at (0,0,0) (relative to env root).
        # Pallet Dims: 120 (Y), 80 (X). 
        # Grid X: 0-15 -> 0-80cm.
        # Grid Y: 0-23 -> 0-120cm.
        # Pos = (Grid * Res) - (Size/2) + (BoxDim/2)
        # Z = Top of heightmap? Or we assume "Drop" means spawn at fixed Height?
        # "Drop & Settle" -> Spawn high, let physics settle.
        
        # Which envs are effectively placing? (Op==0 OR Valid Retrieve)
        active_place_mask = (op == 0) | valid_retrieve
        
        target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        target_rot = torch.zeros((self.num_envs, 4), device=self.device)
        target_rot[:, 0] = 1.0 # Identity
        
        if torch.any(active_place_mask):
            idx_place = env_indices[active_place_mask]
            
            # Grid to Local Pos
            # X
            res = self.cfg.grid_res_cm / 100.0
            pallet_x = self.cfg.pallet_size_cm[1] / 100.0 # 80cm
            pallet_y = self.cfg.pallet_size_cm[0] / 100.0 # 120cm
            
            # Center offset
            x_offset = -pallet_x / 2.0
            y_offset = -pallet_y / 2.0
            
            # Coords
            gx = grid_x[active_place_mask].float()
            gy = grid_y[active_place_mask].float()
            
            tx = x_offset + (gx * res) + (res/2.0) # Center of slot
            ty = y_offset + (gy * res) + (res/2.0)
            
            # Z: Spawn at fixed height above max heightmap?
            # Or retrieve current stack height at that pos?
            # Simplest: Spawn at Z=1.5m (drop).
            tz = torch.full_like(tx, 1.5)
            
            target_pos[idx_place, 0] = tx + self.scene.env_origins[idx_place, 0]
            target_pos[idx_place, 1] = ty + self.scene.env_origins[idx_place, 1]
            target_pos[idx_place, 2] = tz + self.scene.env_origins[idx_place, 2] # Abs World
            
            # Rotation: 0=0deg, 1=90deg
            # Quat for 90 deg around Z: [0.707, 0, 0, 0.707] (Example)
            # 0 deg: [1, 0, 0, 0]
            # 90 deg: [0.7071, 0, 0, 0.7071]
            r_mask = (rot[active_place_mask] == 1)
            # Default identity already set
             # Set 90 deg
            # q = (w, x, y, z) in Isaac Lab order
            target_rot[idx_place[r_mask]] = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=self.device)

        # Apply Teleportation
        # For ALL envs: 
        #   Place/Retrieve: Go to Target
        #   Store: Go to "Holding" (e.g. -10 Z)
        
        final_pos = target_pos.clone()
        final_rot = target_rot.clone()
        
        # Stores (and Invalid Retrieves) -> Teleport Away
        away_mask = ~(active_place_mask)
        if torch.any(away_mask):
            idx_away = env_indices[away_mask]
            final_pos[idx_away, 2] = -10.0 # Under ground
            
        # Write to Sim
        # We need to act on `global_active_idx`
        # But `write_root_pose_to_sim` usually takes ALL indices?
        # NO, we can modify specific indices if the View supports it.
        # RigidObject view usually supports `env_indices` if wrapping all?
        # `write_root_pose_to_sim` takes `pose` (N,7) and `env_ids`?
        # Usually it writes to ALL if no indices specified.
        # `set_world_poses(pos, rot, indices)`
        
        # We need to construct full tensors for the view
        # Or simpler: Update ALL root poses? No, too expensive.
        # Just update the ACTIVE boxes?
        # The view "boxes" might be *all* boxes.
        # We want to update `global_active_idx`.
        
        # Because `global_active_idx` are scattered, we might need a gathered write.
        # Isaac Lab `RigidObject.write_root_pose_to_sim(root_pos, root_rot, env_ids=...)`
        # Wait, if `env_ids` is passed, it updates *all bodies in those envs*?
        # OR does it take indices of the BODIES?
        # Docs: `indices` (Tensor, optional): Indices of the prims to update.
        
        self.scene["boxes"].write_root_pose_to_sim(final_pos, final_rot, indices=global_active_idx)
        
        # Reset velocities
        zeros_vel = torch.zeros_like(final_pos)
        # velocity is 6D (lin + ang)
        zeros_vel6 = torch.zeros((self.num_envs, 6), device=self.device)
        self.scene["boxes"].write_root_velocity_to_sim(zeros_vel6, indices=global_active_idx)

        # 3. Physics Loop
        # Check if we need to run physics
        if torch.any(active_place_mask):
            # 50 Frames
            for _ in range(50):
                self.sim.step(render=False) # Or self.enable_render
                
            # Update buffers after physics
            self.scene.update(dt=self.sim.dt * 50)
            
        # 4. Stability Check & Rewards
        # Compare Final Pose vs Target Pose (Height/Drift)
        
        # Get final poses
        current_pos = self.scene["boxes"].data.root_pos_w[global_active_idx]
        
        # Stability: 
        # Check 1: Did it fall? Z < 0
        # Check 2: Drift? dist(xy) > threshold
        
        # We need Target (x,y) from before.
        # target_pos was Absolute World.
        
        dist = torch.norm(current_pos[:, :2] - target_pos[:, :2], dim=-1)
        z_diff = torch.abs(current_pos[:, 2] - target_pos[:, 2]) # Might be large if it dropped?
        # Actually target_z was drop spawn height (1.5). Final z should be resting.
        # What is expected resting Z? Depends on stack.
        # We just check if it fell off pallet.
        
        # Logic:
        # If Z < 0.1 (Floor): Failed.
        # If Dist > 10cm: Unstable?
        
        # Compute Reward
        rew, dones = self.compute_rewards(op, active_place_mask, valid_retrieve, dist, current_pos)
        
        # Done logic (Max Boxes or Crash)
        # Crash/Fall -> Done
        fell = current_pos[:, 2] < 0.05
        dones = dones | fell
        
        # Advance Box Index
        # Only advance if we Placed or Stored.
        # Retrieve: We "consumed" a buffer item. The `active_box` was used as the vessel.
        # So we effectively used up one physical box slot? Yes.
        self.box_idx += 1
        
        # Max Boxes Done
        dones = dones | (self.box_idx >= self.cfg.max_boxes)
        
        # Aging Buffer
        self.buffer_state[:, :, 4] += 1.0 # Age
        
        # Handle Reset
        reset_ids = env_indices[dones]
        if len(reset_ids) > 0:
            self._reset_idx(reset_ids)
            
        # Observations
        obs = self._get_observations()
        # Add buffer state to obs? 
        # _get_observations returns dict. 
        # "policy" key usually expects flattened vector.
        # Hybrid NN expects separate inputs?
        # User said: "Visual Head... Vector Head... Fusion".
        # We should return dict with keys "visual" and "vector".
        # But `DirectRLEnv` usually flattens to "policy".
        # We'll update `_get_observations` to return structured dict needed by custom model.
        # OR we pack it into "policy" and let model unpack?
        # User said "Vector Head... Input shape (3 + 50) (Current Box dims + Flattened Buffer State)".
        # Plus Proprio? 
        # We need to align with `actor_critic.py` later.
        
        return obs, rew, dones, {}

    def compute_rewards(self, op, place_mask, retrieve_mask, dist, final_pos):
        # Buffer Storage: -0.1
        # Buffer Retrieval: +2.0
        # Buffer Aging: -0.01 * sum(ages)
        # Pallet Success: +1.0 + Volume Bonus
        # Stability Failure: -10.0
        
        rew = torch.zeros(self.num_envs, device=self.device)
        
        # Storage
        store_mask = (op == 1)
        rew[store_mask] -= 0.1
        
        # Retrieval
        rew[retrieve_mask] += 2.0
        
        # Aging
        ages = self.buffer_state[:, :, 4].sum(dim=1)
        rew -= 0.01 * ages
        
        # Pallet Success (Place OR Retrieve that resulted in place)
        # Conditions: Not fell, Stable.
        fell = final_pos[:, 2] < 0.05
        unstable = dist > 0.10 # 10cm drift tolerance
        
        failure = place_mask & (fell | unstable)
        success = place_mask & (~failure)
        
        rew[failure] -= 10.0
        rew[success] += 1.0
        
        # Volume Bonus?
        # Need volume of active box.
        idx = self.box_idx
        dims = self.box_dims_tensor[torch.arange(self.num_envs), idx]
        vol = dims[:, 0] * dims[:, 1] * dims[:, 2]
        
        # Total Volume (approx 2.4 m3?)
        # Just proportional bonus
        rew[success] += vol[success] # Simple
        
        # Return Dones for immediate failures
        dones = failure
        
        return rew, dones

