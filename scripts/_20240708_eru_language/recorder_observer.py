import torch

from .observer_base import ObserverBase

# ---------------------------------------------------------------------------

class RecorderObserver(ObserverBase):

    # -----------------------------------------------------------------------

    def __init__(self):

        super().__init__()
        return

    # -----------------------------------------------------------------------
    # Note: in this experiment, tokens 1 and 2 carry signal; all others are random.

    def observe(self, ctx):

        i_sample = 0 # we examine a single sample in every batch
        x = ctx["batch"]["x"][0]
        is_all_satisfactory_observation_targets = lambda iss: \
            all(len(is_) >= 1 for is_ in iss)

        while i_sample < x.shape[0]:
            x_values = x[i_sample].tolist()
            i1s, i2s = [], []
            for i, v in enumerate(x_values):
                match v: # in this setup, tokens 1 and 2 carry main signal - we track them. Token 3 is irrelevant and is a baseline
                    case 1: i1s.append(i)
                    case 2: i2s.append(i)
            if is_all_satisfactory_observation_targets([i1s, i2s]):
                break
            i_sample += 1

        if is_all_satisfactory_observation_targets([i1s, i2s]):

            i1, i2, iEOS = i1s[0], i2s[0], -1
            c_heads = ctx["self-attention 0"]["keys"][i_sample].shape[0]

            projections_observations_cur = []
            projections_observation_plan = [
                ((1, i1), "self-attention 0", "keys"),
                ((2, i2), "self-attention 0", "keys"),
                (("EOS", iEOS), "self-attention 0", "keys"),
                ((1, i1), "self-attention 0", "queries"),
                ((2, i2), "self-attention 0", "queries"),
                (("EOS", iEOS), "self-attention 0", "queries"),

                ((1, i1), "self-attention 1", "keys"),
                ((2, i2), "self-attention 1", "keys"),
                (("EOS", iEOS), "self-attention 1", "keys"),
                ((1, i1), "self-attention 1", "queries"),
                ((2, i2), "self-attention 1", "queries"),
                (("EOS", iEOS), "self-attention 1", "queries"),
            ]
            for (token, token_i), self_attention_layer, kind in projections_observation_plan:
                for i_head in range(c_heads):
                    projections_observations_cur.append((
                        "|".join([self_attention_layer, kind, str(i_head), str(token)]),
                        ctx[self_attention_layer][kind][i_sample][i_head][token_i].tolist(),
                    ))

            attentions_observations_cur = []
            attentions_observation_plan = [
                ((1, i1), (2, i2), "self-attention 0"),
                ((1, i1), ("median", "median"), "self-attention 0"),
                ((2, i2), (1, i1), "self-attention 0"),
                ((2, i2), ("median", "median"), "self-attention 0"),
                (("EOS", iEOS), ("median", "median"), "self-attention 1"),
                (("EOS", iEOS), (1, i1), "self-attention 0"),
                (("EOS", iEOS), (2, i2), "self-attention 0"),

                ((1, i1), (2, i2), "self-attention 1"),
                ((2, i2), (1, i1), "self-attention 1"),
                (("EOS", iEOS), (1, i1), "self-attention 1"),
                (("EOS", iEOS), (2, i2), "self-attention 1"),
                (("EOS", iEOS), ("median", "median"), "self-attention 1"),
            ]
            for (token_a, token_a_i), (token_b, token_b_i), self_attention_layer in attentions_observation_plan:
                for i_head in range(c_heads):
                    attentions_observations_cur.append((
                        (
                            "|".join([self_attention_layer, str(i_head), str(token_a), str(token_b)]),
                            sorted(ctx[self_attention_layer]["attention-weights"][i_sample][i_head][token_a_i].tolist(), reverse=True) \
                                [ctx[self_attention_layer]["attention-weights"].shape[3] // 2],
                        )
                        if token_b == "median"
                        else (
                            "|".join([self_attention_layer, str(i_head), str(token_a), str(token_b)]),
                            ctx[self_attention_layer]["attention-weights"][i_sample][i_head][token_a_i][token_b_i].item(),
                        )
                    ))

            self.observations.append({
                "i-batch": ctx["batch"]["i-batch"],
                "i-sample": i_sample,
                "x": x_values,
                "projections": projections_observations_cur,
                "attentions": attentions_observations_cur,
            })

        return

    # -----------------------------------------------------------------------

    def finalize(self, params):

        def print_summary(i_observation):

            print(f"observation: {i_observation}")

            attentions = self.observations[i_observation]["attentions"]
            for t in attentions[:4] + attentions[-3:]:
                print(t)

            # print(self.observations[i_observation]["projections"][-1][0])
            level_1_eos_query = self.observations[i_observation]["projections"][-1][1]
            # print(self.observations[i_observation]["projections"][6][0])
            level_1_eos_key_1 = self.observations[i_observation]["projections"][8][1]
            # print(self.observations[i_observation]["projections"][7][0])
            level_1_eos_key_2 = self.observations[i_observation]["projections"][9][1]

            sim = torch.nn.CosineSimilarity(dim=0)
            c_prefix = 8
            print(f"level 1 eos:1 {sim(torch.tensor(level_1_eos_query), torch.tensor(level_1_eos_key_1)):.2f}")
            # print(" ".join(f"{v1:+.1f}|{v2:+.1f}" for v1, v2 in zip(level_1_eos_query[:c_prefix], level_1_eos_key_1[:c_prefix])))
            print(f"level 1 eos:2 {sim(torch.tensor(level_1_eos_query), torch.tensor(level_1_eos_key_2)):.2f}")
            # print(" ".join(f"{v1:+.1f}|{v2:+.1f}" for v1, v2 in zip(level_1_eos_query[:c_prefix], level_1_eos_key_2[:c_prefix])))

            return

        print_summary(-1)

        return
