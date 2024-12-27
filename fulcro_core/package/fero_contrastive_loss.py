import torch

# ---------------------------------------------------------------------------

class FeroContrastiveLoss(torch.nn.Module):

    # -----------------------------------------------------------------------

    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    # -----------------------------------------------------------------------

    def forward(self, dist, label):
        loss = (
            torch.mean(
                1/2 * (label) * torch.pow(dist, 2) +
                1/2 * (1-label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
            )
        )
        return loss