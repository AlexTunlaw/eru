from typing import List, Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------

EOU = 0

# ---------------------------------------------------------------------------

class Token(object):
    pass

# ---------------------------------------------------------------------------

class Class:
    def __init__(self, label, ngram, p):
        self.label = label
        assert not any(token == EOU for token in ngram)
        self.ngram = ngram
        self.p = p

# ---------------------------------------------------------------------------

class EruNgramLanguage:

    # -----------------------------------------------------------------------

    @classmethod
    def make_from_config(cls, config):
        return EruNgramLanguage(
            vocab=list(range(config["vocab-size"])),
            vocab_ps=config["vocab-ps"],
            classes=config["classes"],
            utterance_len=config["utterance-len"],
        )

    # -----------------------------------------------------------------------

    def __init__(self,
        vocab: List[Token],
        vocab_ps: List[float],
        classes: Dict[Token, List[Tuple[float, Token]]],
        utterance_len: int,
    ):
        self.vocab = vocab
        assert vocab_ps[0] == 0.0
        self.vocab_ps = vocab_ps
        self.classes = [
            Class(label=label, ngram=ngram, p=p)
            for label, (p, ngram) in classes.items()
        ]
        star_clss = [
            cls
            for cls in self.classes
            if cls.ngram == "*"
        ]
        assert len(star_clss) == 1
        self.star_cls = star_clss[0]
        self.ps = [cls.p for cls in self.classes]
        assert sum(self.ps) == 1.0
        self.utterance_len = utterance_len
        return

    # -----------------------------------------------------------------------

    def is_class_expressed_in_utterance(self, cls, utterance):

        if cls == self.star_cls:
            return False

        return all(
            token in utterance
            for token in cls.ngram
        )

    # -----------------------------------------------------------------------

    def generate(self):

        max_attemts_to_generate_without_collisions = 100

        cls = np.random.choice(self.classes, p=self.ps)

        for k in range(max_attemts_to_generate_without_collisions):

            # base
            utterance = [
                np.random.choice(self.vocab, p=self.vocab_ps)
                for _k in range(self.utterance_len - 1)
            ] + [EOU]
            assert len(utterance) == self.utterance_len

            # class
            if cls != self.star_cls:

                indices = np.random.choice(
                    list(range(len(utterance) - 1)),
                    size=len(cls.ngram),
                    replace=False, # no duplicates
                )
                assert len(indices) == len(set(indices))
                assert len(indices) == len(cls.ngram)
                for i, token in zip(indices, cls.ngram):
                    utterance[i] = token
                assert self.is_class_expressed_in_utterance(cls, utterance)

            # collision?
            if not any(
                self.is_class_expressed_in_utterance(cls_other, utterance)
                for cls_other in self.classes
                if cls != cls_other
            ):
                break
        assert k < max_attemts_to_generate_without_collisions, f"The language is impossibly dense."

        return utterance, cls.label