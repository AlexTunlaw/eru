import time

from ml3 import CsvFile, CsvSchema, CsvLine, ProgressTrace, AllRandoms

from fero_lib import OaiBuilder

from eru_lib.v1 import (
    EruExampleStream,
    EruSimilarityExampleStream,
    EruGruBinaryClassificationWorkflow,
    EruSelfAttentionBinaryClassificationWorkflow,
    EruSelfAttentionSimilarityWorkflow,
)   

# ---------------------------------------------------------------------------

class EruLanguageBuilder(OaiBuilder):

    # -----------------------------------------------------------------------

    def generate_e1_language_data(self, params):

        training_data_file = self.get_path(params["outputs"]["training-data"])
        (
            utterance_column,
            signature_column,
        ) = training_data_columns = params["training-data-columns"]
        training_data_schema = CsvSchema(columns=training_data_columns)

        example_stream = EruExampleStream.make_from_config(params)

        training_data_lines = []
        progress_trace = ProgressTrace(trace_every_seconds=2, check_every_ticks=10)
        for _k in range(params["training-data-count"]):
            utterance, signature = example_stream.get_example()
            training_data_lines.append(CsvLine(schema=training_data_schema, values={
                utterance_column: utterance,
                signature_column: "_".join(str(k) for k in signature),
            }))
            progress_trace.tick()
        progress_trace.done()
            
        CsvFile.save_to_csv(training_data_file, training_data_lines, training_data_schema)
        return
    
    # -----------------------------------------------------------------------

    def train_e1_gru_binary_classification(self, params):

        AllRandoms.set_random_seed(13)

        example_stream = EruExampleStream.make_from_config({
            **params["language"],
            "mode": "binary-classification-of-root-expression"
        })

        EruGruBinaryClassificationWorkflow.train(
            example_stream=example_stream,
            config=params["training-config"]
        )

        return

    # -----------------------------------------------------------------------

    def train_e1_self_attention_binary_classification(self, params):

        AllRandoms.set_random_seed(13)

        example_stream = EruExampleStream.make_from_config({
            **params["language"],
            "mode": "binary-classification-of-root-expression"
        })

        AllRandoms.set_random_seed(int(1000 * time.time()) % 2**31)

        batch_count = params["training-config"]["batch-count"]
        vocab_token_attention_developments_log = [
            {}
            for _k in range(batch_count)
        ]

        def observe_forward(ctx):
            record_threshold = 0.0
            target = 22
            for k in range(ctx.batch_size):
                for h in range(ctx.c_heads):
                    attention_weights = ctx.attention_weights[k, h, -1].tolist()
                    for i, w in enumerate(attention_weights):
                        if w > record_threshold:
                            vocab_token = int(ctx.x[k, i].item())
                            vocab_token_attention_developments_log[ctx.i_batch][vocab_token] = w
                    continue

        c_runs = int(params["run-count"])
        loss_logs = []
        for i_run in range(c_runs):

            print(f"*** run {i_run}")

            results = EruSelfAttentionBinaryClassificationWorkflow.train(
                example_stream=example_stream,
                config={
                    **params["training-config"],
                    "observe-forward-fn": observe_forward
                }
            )

            loss_logs.append(results["loss-log"])

            continue

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
        CsvFile.save_to_csv(
            self.get_path(params["outputs"]["loss-logs"]),
            output_lines, output_schema
        )

        # vocab attention developments log
        output_schema=CsvSchema(columns=["token"] + [str(k) for k in range(batch_count)])
        output_lines = [
            CsvLine(schema=output_schema, values={
                "token": token,
                **{
                    str(i_batch): ""
                    for i_batch in range(batch_count)
                }
            })
            for token in range(example_stream.language.vocab_distribution.size)
        ]
        for i_batch in range(batch_count):
            for token in range(example_stream.language.vocab_distribution.size):
                w = vocab_token_attention_developments_log[i_batch].get(token)
                if w is not None:
                    output_lines[token][str(i_batch)] = f"{w:.5f}"
        CsvFile.save_to_csv(self.get_path("test-1: csv"), output_lines, output_schema)

        return

    # -----------------------------------------------------------------------

    def train_e1_self_attention_similarity(self, params):

        AllRandoms.set_random_seed(13)

        example_stream = EruSimilarityExampleStream.make_from_config({
            **params["language"],
            "mode": "binary-classification-of-root-expression",
            "example-stream": params["example-stream"]
        })

        AllRandoms.set_random_seed(int(1000 * time.time()) % 2**31)

        c_runs = int(params["run-count"])
        loss_logs = []
        for i_run in range(c_runs):

            print(f"*** run {i_run}")

            results = EruSelfAttentionSimilarityWorkflow.train(
                example_stream=example_stream,
                config=params["training-config"]
            )

            loss_logs.append(results["loss-log"])

            continue

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

        return