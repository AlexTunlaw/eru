import random
import numpy as np

from .fnv_hash import FnvHash

# ---------------------------------------------------------------------------
# Use this to set seed for all randoms - typically desired for 100% reproducible
# training runs

class AllRandoms:

    # -----------------------------------------------------------------------

    idx = 13

    # -----------------------------------------------------------------------
    
    @classmethod
    def set_random_seed(self, seed=13, set_torch=True):
        random.seed(seed)
        np.random.seed(seed)
        if set_torch:
            try:
                import torch
                torch.manual_seed(seed)
            except ImportError:
                pass

    # -----------------------------------------------------------------------
    # Call this method when reproducibility is disrupted by external components
    # (e.g., torch?) - just before the critical path for reproducibility.

    @classmethod
    def next_checkpoint(self):
        self.idx = (self.idx + 1) % pow(2, 15)
        self.set_random_seed(seed=FnvHash.compute_n(self.idx))