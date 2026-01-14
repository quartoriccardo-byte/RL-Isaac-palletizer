
"""
Shim that forwards to the archived legacy script.

Kept for backwards compatibility with old command lines; the supported
pipeline does not use this script anymore.
"""

from legacy.gen_mask_dataset import main


if __name__ == "__main__":
    main()
