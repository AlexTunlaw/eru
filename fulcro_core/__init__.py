import torch

from .package.fulcro_contrastive_loss import FulcroContrastiveLoss
from .package.fulcro_siamese_learner import FulcroSiameseLearner
from .package.fulcro_window_based_loss_level import FulcroWindowBasedLossLevel
from .package.fulcro_siamese_contrastive_workflow import FulcroSiameseContrastiveWorkflow

from .package.streams.fulcro_base_example_stream import FulcroBaseExampleStream
from .package.streams.fulcro_simple_tensor_list_example_stream import FulcroSimpleTensorListExampleStream

# ---------------------------------------------------------------------------

def custom_base_softmax(x, base=2, dim=-1):
    x_exp = torch.pow(base, x)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)
