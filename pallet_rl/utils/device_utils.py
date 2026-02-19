import torch

def pick_supported_cuda_device(min_cc=(7, 5)):
    """
    Selects the first available CUDA device with compute capability >= min_cc.
    Forces PyTorch to use this device and sets it as default.
    
    Args:
        min_cc (tuple): Minimum required compute capability (major, minor).
        
    Returns:
        tuple: (selected_index, device_str) where device_str is "cuda:<index>"
        
    Raises:
        RuntimeError: If no compatible device is found.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system.")

    count = torch.cuda.device_count()
    selected_index = None
    selected_props = None

    print(f"[GPU Selection] scanning {count} devices...")

    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        cc_major = props.major
        cc_minor = props.minor
        mem_gb = props.total_memory / (1024 ** 3)
        
        print(f"  Device {i}: {props.name} (CC {cc_major}.{cc_minor}, {mem_gb:.1f} GB)")

        if selected_index is None:
            # Check if meets requirements
            if cc_major > min_cc[0] or (cc_major == min_cc[0] and cc_minor >= min_cc[1]):
                selected_index = i
                selected_props = props

    if selected_index is None:
        raise RuntimeError(
            f"No CUDA device found with Compute Capability >= {min_cc[0]}.{min_cc[1]}. "
            "Update hardware or adjust requirements."
        )

    # Force the device
    torch.cuda.set_device(selected_index)
    device_str = f"cuda:{selected_index}"
    torch.set_default_device(device_str)

    # Defensive check
    current = torch.cuda.current_device()
    current_props = torch.cuda.get_device_properties(current)
    if current != selected_index:
         raise RuntimeError(f"Failed to set current CUDA device. Expected {selected_index}, got {current}")
         
    if current_props.major < min_cc[0]:
         raise RuntimeError(
            f"Selected GPU {current_props.name} has unsupported compute capability "
            f"{current_props.major}.{current_props.minor}"
        )

    print("\n" + "=" * 41)
    print(" CUDA DEVICE FORCED")
    print(f" GPU: {selected_props.name}")
    print(f" CC : {selected_props.major}.{selected_props.minor}")
    print(f" Index: {selected_index}")
    print("=" * 41)

    return f"cuda:{selected_index}"
