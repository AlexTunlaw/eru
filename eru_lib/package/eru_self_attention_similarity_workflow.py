from .eru_self_attention_similarity_model import EruSelfAttentionSimilarityModel
from .eru_base_similarity_workflow import EruBaseSimilarityWorkflow

# ---------------------------------------------------------------------------

class EruSelfAttentionSimilarityWorkflow(EruBaseSimilarityWorkflow):
    
    # -----------------------------------------------------------------------

    @classmethod
    def make_model(cls, example_stream, config):

        return EruSelfAttentionSimilarityModel(
            vocab_size=cls.BASE + len(example_stream.language.vocab_distribution.items),
            embedding_dim=config["model"]["embedding-dim"],
            attention_dim=config["model"]["attention-dim"],
            c_heads=config["model"]["c-heads"],
            attention_weights_mode=config["model"]["attention-weights-mode"]
        )
