import numpy as np

from .eru_language import EruLanguage

# ---------------------------------------------------------------------------

class EruExampleStream:
    
    # -----------------------------------------------------------------------

    @classmethod
    def make_from_config(cls, config):

        return EruExampleStream(
            language=EruLanguage.make_from_config(config),
            utterance_min_len=config["utterance-min-len"],
            utterance_max_len=config["utterance-max-len"],
            mode=config["mode"]
        )

    # -----------------------------------------------------------------------

    def __init__(self, language, utterance_min_len, utterance_max_len, mode="signature"):

        assert mode in ["signature", "binary-classification-of-root-expression"]

        self.language = language
        self.utterance_min_len = utterance_min_len
        self.utterance_max_len = utterance_max_len
        self.mode = mode

        if self.mode == "binary-classification-of-root-expression":
            self.class_by_root_expression_id = {
                expression.id: np.random.choice(2)
                for expression in self.language.expressions
            }

        return

    # -----------------------------------------------------------------------

    def get_example(self):

        utterance, signature, _tree = self.language.make_utterance(
            utterance_min_len=self.utterance_min_len,
            utterance_max_len=self.utterance_max_len
        )

        match self.mode:
            case "signature":
                annotation = signature
            case "binary-classification-of-root-expression":
                root_expression_id = signature[0]
                annotation = self.class_by_root_expression_id[root_expression_id]

        return utterance, annotation
