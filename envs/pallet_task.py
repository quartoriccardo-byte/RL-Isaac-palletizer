
from typing import Dict
import numpy as np

class MicroSimResult:
    def __init__(self, stable:bool, dcom:float, drot_deg:float, contact_penalty:float):
        self.stable = stable
        self.dcom = dcom
        self.drot_deg = drot_deg
        self.contact_penalty = contact_penalty

def run_micro_sim(sim_iface, thresholds:Dict, T:float, dt:float, substeps:int)->MicroSimResult:
    dcom = np.random.uniform(0.0, thresholds["dcom_m"]*0.5)
    drot = np.random.uniform(0.0, thresholds["drot_deg"]*0.5)
    contact = np.random.uniform(0.0, thresholds["contact_impulse"]*0.5)
    stable = (dcom <= thresholds["dcom_m"] and drot <= thresholds["drot_deg"])
    return MicroSimResult(stable, dcom, drot, contact)

def compute_reward(cfg:Dict, box_vol_frac:float, ms:MicroSimResult, overflow:bool)->float:
    r = cfg["reward"]
    R = r["alpha_volume"] * float(box_vol_frac)
    R += (r["beta_stable"] if ms.stable else -r["beta_stable"])
    R -= r["gamma_contact"] * float(ms.contact_penalty)
    if overflow:
        R -= r["delta_overflow"]
    return R

def compute_heuristics(height_map: np.ndarray, pallet_area: float, max_height: float) -> Dict[str, float]:
    """
    Compute packing heuristics.
    height_map: (L, W) array of heights
    pallet_area: L * W (in same units as height_map cells)
    max_height: maximum allowed height
    """
    # 1. Volume Filling Ratio
    # Volume used = sum(heights) * cell_area (assuming cell_area=1 for ratio if pallet_area is in cells)
    # Total Volume = pallet_area * max_height
    # If height_map is in cm, and max_height in cm.
    
    total_vol = np.sum(height_map)
    max_vol = height_map.size * max_height
    vol_ratio = total_vol / max(1e-6, max_vol)
    
    # 2. Surface Coverage
    # Count non-zero cells
    covered_cells = np.count_nonzero(height_map > 0.01) # epsilon
    coverage = covered_cells / max(1, height_map.size)
    
    # 3. Height Variance
    # Standard deviation of height (only for covered area or total? Usually total surface roughness)
    h_std = np.std(height_map)
    
    return {
        "volume_ratio": float(vol_ratio),
        "surface_coverage": float(coverage),
        "height_std": float(h_std)
    }
