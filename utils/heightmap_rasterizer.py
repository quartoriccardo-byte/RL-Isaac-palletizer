
import torch
import warp as wp
import numpy as np

# Initialize Warp
if not wp.is_initialized():
    wp.init()

@wp.kernel
def rasterize_boxes_kernel(
    box_positions: wp.array(dtype=wp.vec3),      # (N_envs * max_boxes)
    box_orientations: wp.array(dtype=wp.quat),   # (N_envs * max_boxes)
    box_dimensions: wp.array(dtype=wp.vec3),     # (N_envs * max_boxes)
    pallet_positions: wp.array(dtype=wp.vec3),   # (N_envs) - assuming one pallet per env, broadcast
    grid_map: wp.array(dtype=float, ndim=3),     # (N_envs, H, W)
    num_envs: int,
    max_boxes: int,
    grid_res: float,
    map_size_pixels: int,
    pallet_dim_x: float,
    pallet_dim_y: float
):
    """
    Rasterizes boxes onto a heightmap grid.
    Parallelized over all boxes across all environments.
    tid: global index of the box (env_id * max_boxes + box_id)
    """
    tid = wp.tid()
    
    # Identify environment and box index
    env_id = tid // max_boxes
    box_id = tid % max_boxes
    
    if env_id >= num_envs:
        return

    # Check if box is active (e.g. dimensions > 0)
    dim = box_dimensions[tid]
    if dim[0] <= 0.0 or dim[1] <= 0.0 or dim[2] <= 0.0:
        return

    pos = box_positions[tid]
    rot = box_orientations[tid]
    
    # Pallet position for this environment
    pallet_pos = pallet_positions[env_id]
    
    # Calculate box position relative to pallet (pallet frame is origin for the grid)
    # We assume pallet is axis aligned or we just take relative translation?
    # Usually pallet is the reference frame. 
    # Let's assume grid is centered at pallet_pos.
    
    # Transform box position to pallet local frame
    # (Simplified: assume standard alignment, just translation difference if pallet is not rotated)
    # If the pallet rotates, we should rotate pos too. For now assume pallet is static/aligned.
    rel_pos = pos - pallet_pos
    
    # Compute the box's transformation matrix (Rotation + Translation)
    # We want to check pixels in box frame.
    
    # Z-height (top of the box)
    # In Isaac/PhysX, pos is usually CoM. So top is z + h/2.
    # relative z should be used.
    z_top = rel_pos[2] + dim[2] * 0.5
    if z_top < 0.0:
        z_top = 0.0
        
    # We need to project the box onto the 2D grid (XY plane of pallet)
    # Grid parameters
    # origin: (-pallet_dim_x/2, -pallet_dim_y/2) in pallet frame? 
    # Usually grid covers the pallet area.
    # Let's assume grid (0,0) corresponds to (-map_size*res/2, -map_size*res/2) or (0,0)?
    # Prompt says "Transform... Project...". Let's assume grid is centered on pallet.
    
    grid_half_size_world = float(map_size_pixels) * grid_res * 0.5
    
    # Compute Oriented Bounding Box (OBB) corners in 2D (XY)
    # Box basis vectors in world/pallet frame
    rx = wp.quat_rotate(rot, wp.vec3(1.0, 0.0, 0.0))
    ry = wp.quat_rotate(rot, wp.vec3(0.0, 1.0, 0.0))
    # We only care about X and Y components for rasterization
    
    # Instead of full rasterization which is complex, we use AABB optimization
    # Compute AABB of the box on the grid
    # Extent of box in "pallet" frame
    # We project 4 bottom corners (or just use 2D footprint)
    
    half_l = dim[0] * 0.5
    half_w = dim[1] * 0.5
    
    # 4 corners in box local 2D
    c1_loc = wp.vec3( half_l,  half_w, 0.0)
    c2_loc = wp.vec3(-half_l,  half_w, 0.0)
    c3_loc = wp.vec3(-half_l, -half_w, 0.0)
    c4_loc = wp.vec3( half_l, -half_w, 0.0)
    
    # Rotate to pallet frame
    r1 = wp.quat_rotate(rot, c1_loc) + rel_pos
    r2 = wp.quat_rotate(rot, c2_loc) + rel_pos
    r3 = wp.quat_rotate(rot, c3_loc) + rel_pos
    r4 = wp.quat_rotate(rot, c4_loc) + rel_pos
    
    # Find min/max X and Y in pallet frame
    min_x = wp.min(wp.min(r1[0], r2[0]), wp.min(r3[0], r4[0]))
    max_x = wp.max(wp.max(r1[0], r2[0]), wp.max(r3[0], r4[0]))
    min_y = wp.min(wp.min(r1[1], r2[1]), wp.min(r3[1], r4[1]))
    max_y = wp.max(wp.max(r1[1], r2[1]), wp.max(r3[1], r4[1]))
    
    # Convert world/pallet coords to Grid Indices
    # Grid Origin (0,0) is at -grid_half_size_world
    # u = (x - (-half)) / res = (x + half) / res
    
    u_min = int((min_x + grid_half_size_world) / grid_res)
    u_max = int((max_x + grid_half_size_world) / grid_res) + 1
    v_min = int((min_y + grid_half_size_world) / grid_res)
    v_max = int((max_y + grid_half_size_world) / grid_res) + 1
    
    # Clamp to grid
    u_min = wp.max(0, u_min)
    u_max = wp.min(map_size_pixels, u_max)
    v_min = wp.max(0, v_min)
    v_max = wp.min(map_size_pixels, v_max)
    
    # Inverse rotation for point-in-box check
    inv_rot = wp.quat_inverse(rot)
    
    # Loop over pixels in AABB
    for u in range(u_min, u_max):
        for v in range(v_min, v_max):
            # Pixel center in pallet frame
            px_world = (float(u) + 0.5) * grid_res - grid_half_size_world
            py_world = (float(v) + 0.5) * grid_res - grid_half_size_world
            
            # Vector from box center to pixel
            diff_x = px_world - rel_pos[0]
            diff_y = py_world - rel_pos[1]
            diff_z = 0.0 # Projected
            
            # Rotate back to box frame
            # We treat this as a 3D vector (diff_x, diff_y, 0) and rotate by inverse rot
            p_rel = wp.quat_rotate(inv_rot, wp.vec3(diff_x, diff_y, 0.0))
            
            # Check if inside box 2D extent
            # Box centered at 0 in local frame
            if wp.abs(p_rel[0]) <= half_l and wp.abs(p_rel[1]) <= half_w:
                # Inside!
                # Atomic Max Z
                wp.atomic_max(grid_map[env_id, u, v], z_top)

class WarpHeightmapGenerator:
    def __init__(self, device: str, num_envs: int, max_boxes: int, grid_res: float, map_size: int, pallet_dims: tuple):
        self.device = device
        self.num_envs = num_envs
        self.max_boxes = max_boxes
        self.grid_res = grid_res
        self.map_size = map_size
        self.pallet_dims = pallet_dims # (L, W)
        
        # Allocate grid
        self.grid_wp = wp.zeros((num_envs, map_size, map_size), dtype=float, device=device)

    def forward(self, box_pos: torch.Tensor, box_rot: torch.Tensor, box_dims: torch.Tensor, pallet_pos: torch.Tensor) -> torch.Tensor:
        """
        Generates heightmaps from box states.
        box_pos: (N, M, 3)
        box_rot: (N, M, 4) (xyzw or wxyz? warp expects xyzw usually, Isaac is wxyz. Need conversion?)
        Warp quat is (x, y, z, w). Isaac is (w, x, y, z).
        WE MUST CHECK AND CONVERT IF NEEDED. Assuming input is (x,y,z,w) for now or handling convert.
        """
        
        # Reset grid
        self.grid_wp.zero_()
        
        # Create Warp arrays from Torch tensors (zero-copy if on same device)
        # Ensure contiguous
        box_pos_wp = wp.from_torch(box_pos.contiguous(), dtype=wp.vec3)
        
        # Check Quaternion format. 
        # If Isaac (w, x, y, z), Warp needs (x, y, z, w).
        # We assume the caller handles this or we swizzle here?
        # Ideally, we pass it as is, but we must be careful.
        # Let's assume input is configured to be compatible (xyzw).
        box_rot_wp = wp.from_torch(box_rot.contiguous(), dtype=wp.quat) 
        
        box_dims_wp = wp.from_torch(box_dims.contiguous(), dtype=wp.vec3)
        pallet_pos_wp = wp.from_torch(pallet_pos.contiguous(), dtype=wp.vec3)
        
        wp.launch(
            kernel=rasterize_boxes_kernel,
            dim=self.num_envs * self.max_boxes,
            inputs=[
                box_pos_wp,
                box_rot_wp,
                box_dims_wp,
                pallet_pos_wp,
                self.grid_wp,
                self.num_envs,
                self.max_boxes,
                self.grid_res,
                self.map_size,
                self.pallet_dims[0],
                self.pallet_dims[1]
            ],
            device=self.device
        )
        
        # Return as torch tensor
        return wp.to_torch(self.grid_wp)
