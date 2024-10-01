from .eru_gru_binary_classification_model import EruGruBinaryClassificationModel
from .eru_base_binary_classification_workflow import EruBaseBinaryClassificationWorkflow

# ---------------------------------------------------------------------------

class EruGruBinaryClassificationWorkflow(EruBaseBinaryClassificationWorkflow):
    
    # -----------------------------------------------------------------------

    @classmethod
    def make_model(cls, example_stream, config):

        return EruGruBinaryClassificationModel(
            vocab_size=len(example_stream.language.vocab),
            embedding_dim=config["model"]["embedding-dim"],
            gru_hidden_dim=config["model"]["gru-hidden-dim"],
        )
