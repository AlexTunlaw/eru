from typing import Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .observer_base import ObserverBase

# ---------------------------------------------------------------------------

class AttentionLevels01Observer(ObserverBase):

    # -----------------------------------------------------------------------

    def __init__(self, observation_target):

        super().__init__()

        assert observation_target in [
            "layer 0 - EOS to 1, 2",
            "layers 0,1 - 12, 21, EOS"
        ]

        self.observation_target = observation_target
        return

    # -----------------------------------------------------------------------
    # Note: in this experiment, tokens 1 and 2 carry signal; all others are random.
    # We use token 3 as a representative of noise.

    def observe(self, ctx):

        i_sample = 0 # we examine a single sample in every batch
        x = ctx["batch"]["x"][0]
        is_all_satisfactory_observation_targets = lambda iss: \
            all(len(is_) >= 1 for is_ in iss) #TODO what happens with attention when miltiple i_0 and i_1 are present? Does it split?
        
        while i_sample < x.shape[0]:
            x_values = x[i_sample].tolist()
            i1s, i2s = [], []
            for i, v in enumerate(x_values):
                match v: # in this setup, tokens 0 and 1 carry main signal - we track them
                    case 1: i1s.append(i)
                    case 2: i2s.append(i)
            if is_all_satisfactory_observation_targets([i1s, i2s]) >= 1:
                break
            i_sample += 1
    
        if is_all_satisfactory_observation_targets([i1s, i2s]):

            i1, i2, iEOS = i1s[0], i2s[0], -1
            observations_cur = {
                "batch": ctx["batch"]["i-batch"],
            }
            match self.observation_target:
                case "layer 0 - EOS to 1, 2":
                    for token, token_i in list(enumerate([i1, i2])):
                        for self_attention_layer in ["self-attention 0"]:
                            attention_scores = ctx[self_attention_layer]["attention-scores"][i_sample]
                            for i_head in range(attention_scores.shape[0]):
                                observations_cur[f"token {token}, layer {self_attention_layer}, head {i_head}"] = \
                                    attention_scores[i_head][iEOS][token_i].item() # EOS attention to the current token
                case "layers 0,1 - 12, 21, EOS":
                    for (token_a, token_a_i), (token_b, token_b_i), self_attention_layer in [
                        ((1, i1), (2, i2), "self-attention 0"),
                        ((2, i2), (1, i1), "self-attention 0"),
                        (("EOS", iEOS), (1, i1), "self-attention 1"),
                        (("EOS", iEOS), (2, i2), "self-attention 1"),
                    ]:
                        attention_scores = ctx[self_attention_layer]["attention-scores"][i_sample]
                        for i_head in range(attention_scores.shape[0]):
                            observations_cur[f"tokens {token_a}:{token_b}, layer {self_attention_layer}, head {i_head}"] = \
                                attention_scores[i_head][token_a_i][token_b_i].item()
                            
            self.observations.append(observations_cur)

        return

    # -----------------------------------------------------------------------

    def finalize(self, params):

        match self.observation_target:

            case "layer 0 - EOS to 1, 2":
                self.plot_attention_levels(
                    observation_tuples=sum([
                        [
                            (f"token {token}, layer self-attention 0, head {i_head}", label, color)
                            for i_head in range(params["training-config"]["model"]["c-heads"])
                        ]
                        for token, label, color in [
                            (0, "token 1", "blue"),
                            (1, "token 2", "red"),
                        ]
                    ], []),
                )

            case "layers 0,1 - 12, 21, EOS":
                self.plot_attention_levels(
                    observation_tuples=sum([
                        [
                            (f"tokens {tokens}, layer self-attention 0, head {i_head}", label, color)
                            for i_head in range(params["training-config"]["model"]["c-heads"])
                        ]
                        for tokens, label, color in [
                            ("1:2", "tokens L1 1:2", "lightblue"),
                            ("2:1", "tokens L1 2:1", "lightgreen"),
                        ]
                    ] + [
                        [
                            (f"tokens {tokens}, layer self-attention 1, head {i_head}", label, color)
                            for i_head in range(params["training-config"]["model"]["c-heads"])
                        ]
                        for tokens, label, color in [
                            ("EOS:1", "tokens L2 EOS:1", "blue"),
                            ("EOS:2", "tokens L2 EOS:2", "green"),
                        ]
                    ], []),
                )

        return

    # -----------------------------------------------------------------------

    def plot_attention_levels(self,
        observation_tuples: Iterable[Tuple[
            str, # key
            str, # label
            str, # color
        ]]
    ):

        plt.figure(figsize=(10, 6))

        x = [observation["batch"] for observation in self.observations]

        for observation_key, label, color in observation_tuples:

            y = [observation[observation_key] for observation in self.observations]

            plt.plot(
                x, y,
                marker='o', linestyle='--', c=color,
                label=label
            )

        plt.title("Attention levels")
        plt.xlabel("Batch")
        plt.ylabel("Attention")
        plt.legend()

        return
