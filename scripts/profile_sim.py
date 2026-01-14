
"""
Lightweight profiling entrypoint (stub).

The original low-level simulator profiling script has been archived under
`legacy/` and is not part of the supported runtime path. Modern profiling
is expected to be done around `scripts/train.py` using Isaac Lab tools.

This stub avoids ImportError on `legacy.profile_sim` while still giving a
clear message to users who try to run it directly.
"""

from __future__ import annotations


def main() -> None:
    print(
        "[pallet_rl] scripts/profile_sim.py is a stub.\n"
        "Use Isaac Lab's built-in profiling tools around scripts/train.py "
        "for simulator performance analysis."
    )


if __name__ == "__main__":
    main()

