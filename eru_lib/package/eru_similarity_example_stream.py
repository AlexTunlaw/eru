import numpy as np

from .eru_language import EruLanguage
from .eru_example_stream import EruExampleStream

# ---------------------------------------------------------------------------

class EruSimilarityExampleStream:
    
    # -----------------------------------------------------------------------

    @classmethod
    def make_from_config(cls, config):

        return EruSimilarityExampleStream(
            language=EruLanguage.make_from_config(config),
            utterance_min_len=config["utterance-min-len"],
            utterance_max_len=config["utterance-max-len"],
            p_basically_similar=config["example-stream"]["p-basically-similar"],
            similarity_if_basically_similar=config["example-stream"]["similarity-if-basically-similar"],
            expression_vocab_loss=config["example-stream"]["expression-vocab-loss"],
            modifiers_weight=config["example-stream"]["modifiers-weight"],
        )

    # -----------------------------------------------------------------------

    def __init__(self,
        language,
        utterance_min_len,
        utterance_max_len,
        p_basically_similar,
        similarity_if_basically_similar,
        expression_vocab_loss,
        modifiers_weight,
    ):

        self.language = language
        self.utterance_min_len = utterance_min_len
        self.utterance_max_len = utterance_max_len
        self.p_basically_similar = p_basically_similar
        self.similarity_if_basically_similar = similarity_if_basically_similar
        self.expression_vocab_loss = expression_vocab_loss
        self.modifiers_weight = modifiers_weight

        return
    
    # -----------------------------------------------------------------------

    def get_example(self):

        utterance_1, signature_1, tree_1 = self.language.make_utterance(
            utterance_min_len=self.utterance_min_len,
            utterance_max_len=self.utterance_max_len
        )

        if np.random.rand() < self.p_basically_similar:

            utterance_2, _signature_2, tree_2, similarity = tree_1.expression.make_similar_utterance(
                tree_1=tree_1,
                requested_similarity=self.similarity_if_basically_similar,
                expression_vocab_loss=self.expression_vocab_loss,
                modifiers_weight=self.modifiers_weight
            )

        else:

            similarity = 0.0

            while True:
                utterance_2, signature_2, _tree_2 = self.language.make_utterance(
                    utterance_min_len=self.utterance_min_len,
                    utterance_max_len=self.utterance_max_len
                )
                if signature_1[0] != signature_2[0]:
                    break

        return utterance_1, utterance_2, similarity
