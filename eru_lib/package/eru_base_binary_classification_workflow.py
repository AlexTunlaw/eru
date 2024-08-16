import torch

from .eru_base_workflow import EruBaseWorkflow

# ---------------------------------------------------------------------------

class EruBaseBinaryClassificationWorkflow(EruBaseWorkflow):

    # -----------------------------------------------------------------------

    @classmethod
    def make_loss_fn(self):

        return torch.nn.BCELoss()

    # -----------------------------------------------------------------------

    @classmethod
    def make_batch(cls, example_stream, batch_size, max_seq_len):

        assert "Needs to be overridden"

        utterances, labels = [], []
        for _k in range(batch_size):
            utterance, label = example_stream.get_example()
            utterance_normalized = cls.get_normalized_seq(utterance, max_seq_len=max_seq_len)
            utterances.append(torch.tensor(utterance_normalized))
            labels.append(torch.tensor(label, dtype=torch.float))

        x = torch.stack(utterances)
        y = torch.stack(labels)
        return x, y
