
import numpy as np
import os, random, json
from typing import Tuple


class FIFODataset:
    def __init__(self, capacity:int, save_dir:str):
        self.capacity = capacity
        # Masking disabled for Buffer Logic refactor
        # self.data = []
        # self.save_dir = save_dir
        # os.makedirs(save_dir, exist_ok=True)
        pass

    def push(self, state_tensor:np.ndarray, box_props:dict, mask_label:np.ndarray):
        # Masking disabled
        pass

    def sample_batch(self, batch:int)->Tuple[np.ndarray,np.ndarray]:
        # Masking disabled
        return np.array([]), np.array([])

    def __len__(self): return 0

