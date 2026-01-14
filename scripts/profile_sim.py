
"""
Shim that forwards to the archived legacy profiling script.

Modern profiling should be done via Isaac Lab tools around the main
training entrypoint in scripts/train.py.
"""

from legacy.profile_sim import main


if __name__ == "__main__":
    main()

