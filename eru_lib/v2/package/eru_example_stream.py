import numpy as np

from .eru_ngram_language import EruNgramLanguage

# ---------------------------------------------------------------------------

class EruExampleStream:
    
    # -----------------------------------------------------------------------

    @classmethod
    def make_from_config(cls, config):

        return EruExampleStream(
            language=EruNgramLanguage.make_from_config(config),
            mode=config["mode"]
        )

    # -----------------------------------------------------------------------

    def __init__(self, language, mode="signature"):

        assert mode in ["binary-classification"]

        if mode == "binary-classification":
            assert sorted([cls.label for cls in language.classes]) == [0, 1]

        self.language = language
        self.mode = mode
        return

    # -----------------------------------------------------------------------

    def get_example(self):

        match self.mode:
            case "binary-classification":
                utterance, signature = self.language.generate()
                assert 0.0 <= signature <= 1.0
                label = float(signature)

        return utterance, label
