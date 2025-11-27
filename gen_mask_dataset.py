
import os, argparse, numpy as np
from .algo.utils import load_config
from .envs.mask_dataset import FIFODataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--num", type=int, default=20000)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = os.path.join(cfg["train"]["run_dir"], "mask_buffer")
    ds = FIFODataset(cfg["mask"]["fifo_size"], out_dir)

    C = 8 + 5
    L, W = cfg["env"]["grid"]
    for i in range(args.num):
        state = np.random.rand(C, L, W).astype(np.float32)
        mask = (np.random.rand(L, W) > 0.5).astype(np.float32)
        box = {"L": 0.2, "W": 0.1, "H": 0.1, "density": 500.0, "stiffness": 1.0}
        ds.push(state, box, mask)
    print("Dataset written to:", out_dir)

if __name__ == "__main__":
    main()
