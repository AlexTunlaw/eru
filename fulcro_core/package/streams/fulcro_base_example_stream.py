from typing import List
import torch

# ---------------------------------------------------------------------------

class FulcroBaseExampleStream:

    # -----------------------------------------------------------------------

    def get_examples(self, count: int) -> List[torch.Tensor]:
        raise Exception("must be implemented")