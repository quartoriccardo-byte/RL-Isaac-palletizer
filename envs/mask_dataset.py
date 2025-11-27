
import numpy as np
import os, random, json
from typing import Tuple

class FIFODataset:
    def __init__(self, capacity:int, save_dir:str):
        self.capacity = capacity
        self.data = []
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def push(self, state_tensor:np.ndarray, box_props:dict, mask_label:np.ndarray):
        idx = len(self.data)
        np.save(os.path.join(self.save_dir, f"state_{idx}.npy"), state_tensor.astype(np.float32))
        np.save(os.path.join(self.save_dir, f"mask_{idx}.npy"), mask_label.astype(np.float32))
        with open(os.path.join(self.save_dir, f"meta_{idx}.json"), "w") as f:
            json.dump({"box": box_props, "state_shape": list(state_tensor.shape)}, f)
        self.data.append(idx)
        if len(self.data) > self.capacity:
            rm = self.data.pop(0)
            for k in ("state","mask","meta"):
                path = os.path.join(self.save_dir, f"{k}_{rm}.npy") if k!="meta" else os.path.join(self.save_dir, f"{k}_{rm}.json")
                if os.path.exists(path):
                    try: os.remove(path)
                    except: pass

    def sample_batch(self, batch:int)->Tuple[np.ndarray,np.ndarray]:
        import random
        idxs = random.sample(self.data, k=min(batch, len(self.data)))
        X = [np.load(os.path.join(self.save_dir, f"state_{i}.npy")) for i in idxs]
        Y = [np.load(os.path.join(self.save_dir, f"mask_{i}.npy")) for i in idxs]
        return np.stack(X), np.stack(Y)

    def __len__(self): return len(self.data)
