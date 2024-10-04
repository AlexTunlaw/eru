import torch

from .eru_base_workflow import EruBaseWorkflow

# ---------------------------------------------------------------------------

class EruBaseBinaryClassificationWorkflow(EruBaseWorkflow):

    # -----------------------------------------------------------------------

    @classmethod
    def make_model(cls, example_stream, config):

        assert False, "Needs to be overridden"

    # -----------------------------------------------------------------------

    @classmethod
    def make_loss_fn(self):

        return torch.nn.BCELoss()

    # -----------------------------------------------------------------------

    @classmethod
    def make_batch(cls, example_stream, batch_size, max_seq_len):

        utterances, labels = [], []
        for _k in range(batch_size):
            utterance, label = example_stream.get_example()
            assert 0.0 <= label <= 1.0
            utterances.append(torch.tensor(utterance))
            labels.append(torch.tensor(label, dtype=torch.float))

        x = torch.stack(utterances)
        y = torch.stack(labels)
        return x, y

    # -----------------------------------------------------------------------

    @classmethod
    def get_workflow_metric(cls, y_predicted: torch.Tensor, y_labels: torch.Tensor):

        # (torch version of this would be more concise, but this is nice when debugging)

        y_predicted = y_predicted.tolist()
        y_labels = y_labels.tolist()

        assert len(y_predicted) == len(y_labels)

        y_predicted_snapped = [
            (1.0 if v > 0.5 else 0.0)
            for v in y_predicted
        ]

        c_correct = sum(1 for v1, v2 in zip(y_predicted_snapped, y_labels) if v1 == v2)

        return f"accuracy: {c_correct/len(y_labels)} (={c_correct}/{len(y_labels)})"
    