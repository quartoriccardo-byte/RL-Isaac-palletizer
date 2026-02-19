import torch
import sys

def main():
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        sys.exit(1)

    count = torch.cuda.device_count()
    print(f"CUDA Devices: {count}")
    
    overall_status = True
    
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        print(f"Device {i}: {props.name} (CC {props.major}.{props.minor}, {props.total_memory / 1e9:.1f} GB)")
        
        try:
            # Force device context
            with torch.cuda.device(i):
                t = torch.arange(1024, device=f"cuda:{i}")
                # Simple matmul to test kernel execution
                a = torch.randn(100, 100, device=f"cuda:{i}")
                b = torch.randn(100, 100, device=f"cuda:{i}")
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
            print(f"  Tensor alloc & compute: OK")
        except Exception as e:
            print(f"  Tensor alloc & compute: FAIL ({e})")
            overall_status = False

    if not overall_status:
        sys.exit(1)

if __name__ == "__main__":
    main()
