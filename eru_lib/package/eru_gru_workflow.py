from .eru_gru_model import EruGruModel
from .eru_base_workflow import EruBaseWorkflow

# ---------------------------------------------------------------------------

class EruGruWorkflow(EruBaseWorkflow):
    
    # -----------------------------------------------------------------------

    @classmethod
    def make_model(cls, example_stream, config):

        return EruGruModel(
            vocab_size=cls.BASE + len(example_stream.language.vocab_distribution.items),
            embedding_dim=config["model"]["embedding-dim"],
            gru_hidden_dim=config["model"]["gru-hidden-dim"],
        )
