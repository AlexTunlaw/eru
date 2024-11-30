from typing import Tuple, Iterable
import time

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

    def train_e2_self_attention_binary_classification(self, params):

        AllRandoms.set_random_seed(777)

        example_stream = EruExampleStream.make_from_config({
            **params["language"],
            "mode": "binary-classification"
        })

        AllRandoms.set_random_seed(int(1000 * time.time()) % 2**31)

        #
        # training runs
        #

        c_runs = int(params["run-count"])
        loss_logs = []
        for i_run in range(c_runs):

            observer = params.get("make-observer-fn", lambda: None)()
            
            print(f"*** run {i_run}")

            results = EruSelfAttentionBinaryClassificationWorkflow.train(
                example_stream=example_stream,
                config=params["training-config"],
                observe_fn=lambda ctx: observer.observe(ctx)
            )

            loss_logs.append(results["loss-log"])

            if 0 <= i_run and observer is not None:
                observer.finalize(params)

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
