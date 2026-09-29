"""Determinism controls applied before executing any computational stage."""
import os,random

def enforce(seed=20260702):
 if os.environ.get('PYTHONHASHSEED')!='0':raise RuntimeError('Stages must start through the controller with PYTHONHASHSEED=0.')
 import numpy as np
 random.seed(seed);np.random.seed(seed)
 import torch
 torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.use_deterministic_algorithms(True,warn_only=False)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.set_num_threads(1)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:pass
