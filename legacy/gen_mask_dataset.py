"""
Legacy mask dataset generation script.

This script corresponds to an older U-Net + spatial policy pipeline and is
not part of the canonical Isaac Lab + RSL-RL training loop.

It is kept only for reference and is not expected to run without the legacy
modules that originally accompanied it.
"""

import os
import argparse
import numpy as np

from pallet_rl.algo.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="pallet_rl/configs/base.yaml")
    parser.add_argument("--num", type=int, default=20000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = os.path.join(cfg.get("train", {}).get("run_dir", "runs"), "mask_buffer")

    os.makedirs(out_dir, exist_ok=True)

    # This code path depends on legacy FIFODataset and env.grid layout, which
    # are not present in the modern pipeline; we only keep the structure as
    # documentation of the original intent.
    print(
        "legacy/gen_mask_dataset.py is a stub preserved for reference.\n"
        "It refers to a legacy FIFODataset and mask-based U-Net policy that are "
        "no longer part of the supported training pipeline."
    )


if __name__ == "__main__":
    main()

