from typing import Iterable, Optional
import json

from openai.types.fine_tuning import job_create_params

# ---------------------------------------------------------------------------

class SingleTurnFineTuningExample:

    # -----------------------------------------------------------------------

    def __init__(self, input: str, expected_output: str):

        self.input = input
        self.expected_output = expected_output

    # -----------------------------------------------------------------------

    def get_json_form(self, system_prompt: str):
        return {
            "messages": [
                { "role": "system", "content": system_prompt},
                { "role": "user", "content": self.input },
                { "role": "assistant", "content": self.expected_output }
            ]
        }

    # -----------------------------------------------------------------------

    def get_serialized_form(self, system_prompt: str):
        return json.dumps(self.get_json_form(system_prompt))

# ---------------------------------------------------------------------------

class FeroOaiFineTuningHelpers:

    # -----------------------------------------------------------------------

    SingleTurnFineTuningExample = SingleTurnFineTuningExample

    # -----------------------------------------------------------------------

    @classmethod
    def save_fine_tuning_file(cls, file_name, serialized_examples: Iterable[str]):

        with open(file_name, "w") as f:
            f.writelines([example + "\n" for example in serialized_examples])

    # -----------------------------------------------------------------------

    @classmethod
    def upload_fine_tuning_data_to_oai(cls, local_file_path, openai_client):

        # openai_client.files.list()
        # openai_client.File.delete("..ID..")

        with open(local_file_path, "rb") as f:
            file = openai_client.files.create(
                file=f,
                purpose='fine-tune',
            )
        print(f"OAI file id: {file.id}")
        return

    # -----------------------------------------------------------------------

    @classmethod
    def start_fine_tuning_job(cls,
        uploaded_train_file_id: str,
        base_model: str,
        n_epochs: Optional[int],
        openai_client,
        batch_size="auto",
        uploaded_validate_file_id: str=None,
    ):

        if n_epochs is None:
            n_epochs = 3

        hyper_parameters = job_create_params.Hyperparameters(n_epochs=n_epochs, batch_size=batch_size)

        openai_client.fine_tuning.jobs.create(
            model=base_model,
            training_file=uploaded_train_file_id,
            validation_file=uploaded_validate_file_id,
            hyperparameters=hyper_parameters
        )
        return

    # -----------------------------------------------------------------------

    @classmethod
    def print_status_of_fine_tuning_job(cls, uploaded_file_id: str, openai_client):

        for job in openai_client.fine_tuning.jobs.list(limit=10).data:
            if job.training_file == uploaded_file_id:
                print(job)
        return
