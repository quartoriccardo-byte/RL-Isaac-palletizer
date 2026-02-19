import torch
import sys

def force_supported_cuda_device(min_cc_major=7, min_cc_minor=5):
    """
    Scans available CUDA devices and selects the first one that meets
    the minimum compute capability requirements (default 7.5 for PyTorch 2.7+).

    Returns:
        str: The selected device string (e.g., "cuda:0")
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA devices available found.")

    count = torch.cuda.device_count()
    selected_index = -1
    selected_props = None

    print(f"[INFO] Scanning {count} CUDA devices for CC >= {min_cc_major}.{min_cc_minor}...")

    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {props.name} (CC {props.major}.{props.minor})")
        
        if props.major > min_cc_major or (props.major == min_cc_major and props.minor >= min_cc_minor):
            selected_index = i
            selected_props = props
            break
    
    if selected_index < 0:
        raise RuntimeError(
            f"No GPU found with Compute Capability >= {min_cc_major}.{min_cc_minor}. "
            "Cannot run on this system."
        )

    # Force the device
    torch.cuda.set_device(selected_index)
    torch.set_default_device(f"cuda:{selected_index}")
    
    # Set backend flags
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    
    # Defensive check
    current = torch.cuda.current_device()
    if current != selected_index:
        # This might happen if visible devices interact weirdly, but set_device should handle it
        print(f"[WARN] torch.cuda.current_device() reported {current}, expected {selected_index}")

    # Banner
    print("=" * 41)
    print(" CUDA DEVICE FORCED")
    print(f" GPU: {selected_props.name}")
    print(f" CC : {selected_props.major}.{selected_props.minor}")
    print(f" Index: {selected_index}")
    print("=" * 41)

    return f"cuda:{selected_index}"
