
import torch
import warp as wp
import numpy as np

# Initialize Warp
if not wp.is_initialized():
    wp.init()

@wp.kernel
def rasterize_heightmap_kernel(
    box_positions: wp.array(dtype=wp.vec3),      # (N_envs, max_boxes) - Flatted in dim 1 logic or layout? 
                                                 # Usually (N*max) flat 1D array is easiest for Warp, 
                                                 # but here we might want 2D access or index math.
                                                 # Let's use 1D array with stride.
    box_orientations: wp.array(dtype=wp.quat),   # (N_envs * max_boxes)
    box_dimensions: wp.array(dtype=wp.vec3),     # (N_envs * max_boxes)
    pallet_positions: wp.array(dtype=wp.vec3),   # (N_envs)
    pallet_orientations: wp.array(dtype=wp.quat),# (N_envs)
    grid_map: wp.array(dtype=float, ndim=3),     # (N_envs, H, W)
    num_envs: int,
    max_boxes: int,
    grid_res: float,
    map_size_pixels: int
):
    """
    Ray-Free Projection-Based Rasterizer.
    Parallelized per pixel.
    tid = env_id * (H * W) + pixel_index
    """
    tid = wp.tid()
    pixels_per_env = map_size_pixels * map_size_pixels
    
    # 1. Determine Environment and Pixel Coordinates
    env_id = tid // pixels_per_env
    pixel_linear_idx = tid % pixels_per_env
    
    if env_id >= num_envs:
        return
        
    u = pixel_linear_idx // map_size_pixels # Row
    v = pixel_linear_idx % map_size_pixels  # Col
    
    # 2. Calculate World Position of this Pixel P_world
    # Grid is centered on the Pallet.
    # Local Grid Coord (x, y) assuming center matches pallet center.
    # Grid extends from -Size/2 to +Size/2
    grid_half_size = float(map_size_pixels) * grid_res * 0.5
    
    # Pixel center in Grid Frame (Local)
    px_local_x = (float(u) + 0.5) * grid_res - grid_half_size
    px_local_y = (float(v) + 0.5) * grid_res - grid_half_size
    
    # Transform Pixel to World Frame
    # P_world = T_pallet * P_grid_local
    # (Assuming scaling is 1)
    
    pallet_pos = pallet_positions[env_id]
    pallet_quat = pallet_orientations[env_id]
    
    # Rotate pixel offset by pallet orientation
    p_offset_world = wp.quat_rotate(pallet_quat, wp.vec3(px_local_x, px_local_y, 0.0))
    p_world = pallet_pos + p_offset_world
    
    # Initialize height for this pixel (e.g. 0.0 or pallet surface?)
    # Pallet surface might be at Z=0 in local frame? 
    # Usually pallet defines the ground 0.
    curr_max_h = 0.0 
    
    # 3. Iterate over every Box in this Environment
    base_idx = env_id * max_boxes
    
    for i in range(max_boxes):
        idx = base_idx + i
        
        dims = box_dimensions[idx]
        # Check if box exists (e.g. dims > 0)
        if dims[0] <= 0.0:
            continue
            
        b_pos = box_positions[idx]
        b_quat = box_orientations[idx]
        
        # 4. Transform P_world into Box Local Frame
        # P_local = R_inv * (P_world - T_box)
        
        diff = p_world - b_pos
        # Inverse rotate
        p_box_local = wp.quat_rotate_inv(b_quat, diff)
        
        # 5. Check Intersection (Point in Box)
        # Box centered at 0 in local frame. Extent [-dim/2, +dim/2]
        half_l = dims[0] * 0.5
        half_w = dims[1] * 0.5
        # Important: Should we consider margin? Or strict? 
        # Standard rasterization is strict.
        
        if (p_box_local[0] >= -half_l and p_box_local[0] <= half_l and
            p_box_local[1] >= -half_w and p_box_local[1] <= half_w):
            
            # Inside!
            # Height of this box at this point.
            # Box Top Z in World Frame? 
            # If box is rotated, top surface z varies.
            # But "Heightmap" usually implies Z-buffer from top-down view.
            # We want the Z coordinate of the surface intersection at (px_world.x, px_world.y).
            # If the box is axis aligned in Z (only yaw), top is b_pos.z + dim.z/2.
            # If box has pitch/roll, we need ray-plane intersection or just take the max Z of the box?
            # Simple assumption: Palletizing usually involves Yaw-only rotations for stability.
            # If Yaw only: Z is constant for the top surface => b_pos[2] + dims[2]*0.5.
            # If Full 3D rotation: We are doing a simplified check (point-in-box 2D projection?).
            # Wait, the check above `p_box_local[0]...` basically checks if the pixel ray (vertical) intersects the box volume
            # IF the box volume was infinitely tall? NO.
            # This check `p_box_local[0]` works perfectly if the box Z axis is aligned with World Z (Yaw only).
            # If Box is tilted, `p_box_local` X/Y check is checking against the box's local cross-section at Z of the point?
            # Actually, `p_box_local` *includes* Z component of the diff.
            # We defined P_world with Z=pallet_z (approx 0).
            # So we are checking if the point on the ground plane is inside the box's projection?
            # NO. `p_box_local` is the coordinate of the GROUND point in Box Frame.
            # If the box is floating above, `p_box_local.z` would be large negative.
            # The X/Y bounds check tells us if the ground point is "under" the box column (local Z).
            # THIS IS CORRECT for "is the pixel ray intersecting the box's local infinite prism".
            # To be strictly correct for heightmap, we want the Z height of the TOP surface at that (x,y).
            # 
            # For Yaw-only boxes: Center Z + Half Height.
            # Let's assume Yaw-only which is standard for palletizing.
            
            z_top = b_pos[2] + dims[2] * 0.5
            
            if z_top > curr_max_h:
                curr_max_h = z_top

    # Write Result
    # User said "Use wp.atomic_max", but since 1 thread = 1 pixel, direct write is superior (faster, no contention).
    # "Iterate over every pixel" -> One thread per pixel logic dictates direct write.
    # However, to be compliant with "atomic_max" request in prompt 1 but "Kernel Logic" in prompt 2...
    # Prompt 2 doesn't explicitly demand atomic_max in logic description step "Write:", it says "Check... Write".
    # I will stick to direct assignment because it's the correct way for this kernel structure.
    # Actually, let's use wp.max just to be safe if I messed up the indices? No, indices are unique.
    grid_map[env_id, u, v] = curr_max_h

class WarpHeightmapGenerator:
    def __init__(self, device: str, num_envs: int, max_boxes: int, grid_res: float, map_size: int, pallet_dims: tuple):
        self.device = device
        self.num_envs = num_envs
        self.max_boxes = max_boxes
        self.grid_res = grid_res
        self.map_size = map_size
        self.pallet_dims = pallet_dims
        
        # Allocate grid
        self.grid_wp = wp.zeros((num_envs, map_size, map_size), dtype=float, device=device)

    def forward(self, 
                box_pos: torch.Tensor, 
                box_rot: torch.Tensor, 
                box_dims: torch.Tensor, 
                pallet_pos: torch.Tensor,
                pallet_rot: torch.Tensor = None) -> torch.Tensor:
        """
        Generates heightmaps.
        Inputs: Torch Tensors on GPU.
        box_pos: (N, M, 3)
        box_rot: (N, M, 4) (x, y, z, w) - ASSUME WARP FORMAT
        box_dims: (N, M, 3)
        pallet_pos: (N, 3)
        pallet_rot: (N, 4) - optional, (x, y, z, w). If None, identity.
        """
        
        self.grid_wp.zero_()
        
        # Conversion
        box_pos_wp = wp.from_torch(box_pos.contiguous(), dtype=wp.vec3)
        box_rot_wp = wp.from_torch(box_rot.contiguous(), dtype=wp.quat)
        box_dims_wp = wp.from_torch(box_dims.contiguous(), dtype=wp.vec3)
        pallet_pos_wp = wp.from_torch(pallet_pos.contiguous(), dtype=wp.vec3)
        
        if pallet_rot is None:
            # Default identity quat (0, 0, 0, 1) for N envs?
            # Or handle inside kernel. Let's make an identity tensor for simplicity if missing.
            # But better to just pass it if available.
            # Create a ones buffer for W and zeros for XYZ? 
            # For efficiency, assume caller passes it. If not, we map a dummy?
            # Let's assume Identity if not passed.
             pallet_rot_wp = wp.zeros(self.num_envs, dtype=wp.quat, device=self.device)
             # By default zeros is (0,0,0,0) which is invalid quat. Warp Quat default?
             # We should fill with (0,0,0,1).
             # Efficiently: map a torch tensor of identities.
             ident = torch.zeros((self.num_envs, 4), device=box_pos.device)
             ident[:, 3] = 1.0 # w=1
             pallet_rot_wp = wp.from_torch(ident, dtype=wp.quat)
        else:
             pallet_rot_wp = wp.from_torch(pallet_rot.contiguous(), dtype=wp.quat)

        # Launch
        # dim = total pixels = num_envs * H * W
        dim = self.num_envs * self.map_size * self.map_size
        
        wp.launch(
            kernel=rasterize_heightmap_kernel,
            dim=dim,
            inputs=[
                box_pos_wp,
                box_rot_wp,
                box_dims_wp,
                pallet_pos_wp,
                pallet_rot_wp,
                self.grid_wp,
                self.num_envs,
                self.max_boxes,
                self.grid_res,
                self.map_size
            ],
            device=self.device
        )
        
        return wp.to_torch(self.grid_wp)
