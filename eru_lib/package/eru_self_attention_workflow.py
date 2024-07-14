from .eru_self_attention_model import EruSelfAttentionModel
from .eru_base_workflow import EruBaseWorkflow

# ---------------------------------------------------------------------------

class EruSelfAttentionWorkflow(EruBaseWorkflow):
    
    # -----------------------------------------------------------------------

    @classmethod
    def make_model(cls, example_stream, config):

        return EruSelfAttentionModel(
            vocab_size=cls.BASE + len(example_stream.language.vocab_distribution.items),
            embedding_dim=config["model"]["embedding-dim"],
            attention_dim=config["model"]["attention-dim"],
            c_heads=config["model"]["c-heads"],
            attention_weights_mode=config["model"]["attention-weights-mode"]
        )
