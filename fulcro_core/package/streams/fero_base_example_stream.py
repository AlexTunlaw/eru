from typing import List
import torch

# ---------------------------------------------------------------------------

class FeroBaseExampleStream:

    # -----------------------------------------------------------------------

    def get_examples(self, count: int) -> List[torch.Tensor]:
        raise Exception("must be implemented")