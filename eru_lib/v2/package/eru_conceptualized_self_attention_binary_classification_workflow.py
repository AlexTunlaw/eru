from .eru_conceptualized_self_attention_binary_classification_model import EruConceptualizedSelfAttentionBinaryClassificationModel
from .eru_base_binary_classification_workflow import EruBaseBinaryClassificationWorkflow

# ---------------------------------------------------------------------------

class EruConceptualizedSelfAttentionBinaryClassificationWorkflow(EruBaseBinaryClassificationWorkflow):
    
    # -----------------------------------------------------------------------

    @classmethod
    def make_model(cls, example_stream, config):

        return EruConceptualizedSelfAttentionBinaryClassificationModel(
            vocab_size=len(example_stream.language.vocab),
            embedding_dim=config["model"]["embedding-dim"],
            c_conceptualizations=config["model"]["c-conceptualizations"],
            c_heads=config["model"]["c-heads"],
            c_layers=config["model"]["c-layers"],
        )
