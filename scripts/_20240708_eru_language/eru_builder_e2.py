from ml3 import CsvFile, CsvSchema, CsvLine, ProgressTrace, AllRandoms

from fero_lib import OaiBuilder

from eru_lib.v2 import (
    EruNgramLanguage,
    EruExampleStream,
    EruGruBinaryClassificationWorkflow,
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
