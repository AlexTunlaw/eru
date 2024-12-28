from typing import List
import numpy as np

import torch

from .fulcro_base_example_stream import FulcroBaseExampleStream

# ---------------------------------------------------------------------------

class FulcroSimpleTensorListExampleStream(FulcroBaseExampleStream):

    # -----------------------------------------------------------------------

    def __init__(self, example_tensors: List[torch.Tensor]):
        
        assert isinstance(example_tensors, list)
        assert all(isinstance(item, torch.Tensor) for item in example_tensors)

        self.example_tensors = example_tensors

    # -----------------------------------------------------------------------

    def get_examples(self, count: int) -> List[torch.Tensor]:

        selected_indices = np.random.choice(len(self.example_tensors), count, replace=True)
        return [self.example_tensors[k] for k in selected_indices]