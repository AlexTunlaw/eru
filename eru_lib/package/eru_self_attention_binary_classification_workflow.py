from .eru_self_attention_binary_classification_model import EruSelfAttentionBinaryClassificationModel
from .eru_base_binary_classification_workflow import EruBaseBinaryClassificationWorkflow

# ---------------------------------------------------------------------------

class EruSelfAttentionBinaryClassificationWorkflow(EruBaseBinaryClassificationWorkflow):
    
    # -----------------------------------------------------------------------

    @classmethod
    def make_model(cls, example_stream, config):

        return EruSelfAttentionBinaryClassificationModel(
            vocab_size=cls.BASE + len(example_stream.language.vocab_distribution.items),
            embedding_dim=config["model"]["embedding-dim"],
            attention_dim=config["model"]["attention-dim"],
            c_heads=config["model"]["c-heads"],
            attention_weights_mode=config["model"]["attention-weights-mode"]
        )
