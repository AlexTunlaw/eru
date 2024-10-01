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
