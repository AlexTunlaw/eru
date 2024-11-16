import time

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from ml3 import CsvFile, CsvSchema, CsvLine, ProgressTrace, AllRandoms

from fero_lib import OaiBuilder

from eru_lib.v2 import (
    EruNgramLanguage,
    EruExampleStream,
    EruGruBinaryClassificationWorkflow,
    EruSelfAttentionBinaryClassificationWorkflow,
)   

# ---------------------------------------------------------------------------

class EruBuilderE2(OaiBuilder):

    # -----------------------------------------------------------------------

    def generate_e2_language_data(self, params):

        AllRandoms.set_random_seed(777)

        language = EruNgramLanguage.make_from_config(params["language"])

        output_file = self.get_path(params["output"])
        (
            utterance_column,
            signature_column,
        ) = output_columns = params["output-columns"]
        output_schema = CsvSchema(columns=output_columns)

        output_lines = []
        for _k in range(params["output-count"]):
            utterance, signature = language.generate()
            output_lines.append(CsvLine(schema=output_schema, values={
                utterance_column: utterance,
                signature_column: signature,
            }))

        CsvFile.save_to_csv(output_file, output_lines, output_schema)
        return

    # -----------------------------------------------------------------------

    def train_e2_gru_binary_classification(self, params):

        AllRandoms.set_random_seed(777)

        example_stream = EruExampleStream.make_from_config({
            **params["language"],
            "mode": "binary-classification"
        })

        EruGruBinaryClassificationWorkflow.train(
            example_stream=example_stream,
            config=params["training-config"]
        )

        return

    # -----------------------------------------------------------------------

    def plot_attention_levels(self, observations, observation_tuples):

        plt.figure(figsize=(10, 6))

        x = [observation["batch"] for observation in observations]

        for observation_key, label, color in observation_tuples:
            y = [observation[observation_key] for observation in observations]

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

    # -----------------------------------------------------------------------

    def train_e2_self_attention_binary_classification(self, params):

        AllRandoms.set_random_seed(777)

        example_stream = EruExampleStream.make_from_config({
            **params["language"],
            "mode": "binary-classification"
        })

        AllRandoms.set_random_seed(int(1000 * time.time()) % 2**31)

        #
        # observations
        #

        # observation_mode = "layer 0 - EOS to 0, 1, 2"
        observation_mode = "layers 0,1 - 01, 10, 02"

        # Note: in this experiment, tokens 0 and 1 carry signal; all others are random.
        # We use token 2 as a representative of noise.
        def observe(ctx, observations):
            i_sample = 0 # we sample a single example in every batch
            x = ctx["batch"]["x"][0]
            is_all_single_observation_targets = lambda iss: \
                all(len(is_) == 1 for is_ in iss) #TODO what happens with attention when miltiple i_0 and i_1 are present? Does it split?
            while i_sample < x.shape[0]:
                x_values = x[i_sample].tolist()
                i0s, i1s, i2s = [], [], []
                for i, v in enumerate(x_values):
                    match v: # in this setup, tokens 0 and 1 carry main signal - we track them
                        case 0: i0s.append(i)
                        case 1: i1s.append(i)
                        case 2: i2s.append(i)
                if is_all_single_observation_targets([i0s, i1s]) and len(i2s) >= 1:
                    break
                i_sample += 1
            if is_all_single_observation_targets([i0s, i1s]) and len(i2s) >= 1:
                i0, i1, i2, iEOS = i0s[0], i1s[0], i2s[0], -1
                observations_cur = {
                    "batch": ctx["batch"]["i-batch"],
                }
                if observation_mode == "layer 0 - EOS to 0, 1, 2":
                    for token, token_i in list(enumerate([i0, i1, i2])):
                        for self_attention_layer in ["self-attention 0"]:
                            attention_scores = ctx[self_attention_layer]["attention-scores"][i_sample]
                            for i_head in range(attention_scores.shape[0]):
                                observations_cur[f"token {token}, layer {self_attention_layer}, head {i_head}"] = \
                                    attention_scores[i_head][iEOS][token_i].item() # EOS attention to the current token
                if observation_mode == "layers 0,1 - 01, 10, 02":
                    for (token_a, token_a_i), (token_b, token_b_i), self_attention_layer in [
                        ((0, i0), (1, i1), "self-attention 0"),
                        ((1, i1), (0, i0), "self-attention 0"),
                        ((0, i0), (2, i2), "self-attention 0"),
                        (("EOS", iEOS), (0, i0), "self-attention 1"),
                        (("EOS", iEOS), (1, i1), "self-attention 1"),
                        (("EOS", iEOS), (2, i2), "self-attention 1"),
                    ]:
                        attention_scores = ctx[self_attention_layer]["attention-scores"][i_sample]
                        for i_head in range(attention_scores.shape[0]):
                            observations_cur[f"tokens {token_a}:{token_b}, layer {self_attention_layer}, head {i_head}"] = \
                                attention_scores[i_head][token_a_i][token_b_i].item()
                observations.append(observations_cur)
            return

        #
        # training runs
        #

        c_runs = int(params["run-count"])
        loss_logs = []
        for i_run in range(c_runs):
            observations = []
            print(f"*** run {i_run}")

            results = EruSelfAttentionBinaryClassificationWorkflow.train(
                example_stream=example_stream,
                config=params["training-config"],
                observe_fn=lambda ctx: observe(ctx, observations=observations)
            )

            loss_logs.append(results["loss-log"])

            if 0 <= i_run:
                match observation_mode:
                    case "layer 0 - EOS to 0, 1, 2":
                        self.plot_attention_levels(
                            observations=observations,
                            observation_tuples=sum([
                                [
                                    (f"token {token}, layer self-attention 0, head {i_head}", label, color)
                                    for i_head in range(params["training-config"]["model"]["c-heads"])
                                ]
                                for token, label, color in [
                                    (0, "token 0", "blue"),
                                    (1, "token 1", "red"),
                                    (2, "token 2", "black")
                                ]
                            ], []),
                        )
                    case "layers 0,1 - 01, 10, 02":
                        self.plot_attention_levels(
                            observations=observations,
                            observation_tuples=sum([
                                [
                                    (f"tokens {tokens}, layer self-attention 0, head {i_head}", label, color)
                                    for i_head in range(params["training-config"]["model"]["c-heads"])
                                ]
                                for tokens, label, color in [
                                    ("0:2", "tokens L1 0:2", "bisque"),
                                    ("0:1", "tokens L1 0:1", "lightblue"),
                                    ("1:0", "tokens L1 1:0", "lightgreen"),
                                ]
                            ] + [
                                [
                                    (f"tokens {tokens}, layer self-attention 1, head {i_head}", label, color)
                                    for i_head in range(params["training-config"]["model"]["c-heads"])
                                ]
                                for tokens, label, color in [
                                    ("EOS:2", "tokens L2 EOS:2", "grey"),
                                    ("EOS:0", "tokens L2 EOS:0", "blue"),
                                    ("EOS:1", "tokens L2 EOS:1", "green"),
                                ]
                            ], []),
                        )
            continue # to next run

        # steps to convergence average
        last_k = 5
        loss_logs_accepted = [
            log
            for log in loss_logs
            if sum(log[-last_k : ]) / last_k < 1.0
        ]
        steps_to_convergence_average = \
            sum(len(log) for log in loss_logs_accepted) / len(loss_logs_accepted)
        print(f"steps to convergence, average: {steps_to_convergence_average}")

        # loss logs
        output_schema=CsvSchema(columns=[f"run-{k}" for k in range(c_runs)])
        longest_run_batches = max(len(log) for log in loss_logs)
        output_lines = [
            CsvLine(schema=output_schema, values={})
            for _k in range(longest_run_batches)
        ]
        for i_run, log in enumerate(loss_logs):
            for k, v in enumerate(log):
                output_lines[k][f"run-{i_run}"] = f"{v:.4f}"

        if "loss-logs" in params["outputs"]:
            CsvFile.save_to_csv(
                self.get_path(params["outputs"]["loss-logs"]),
                output_lines, output_schema
            )

        return
