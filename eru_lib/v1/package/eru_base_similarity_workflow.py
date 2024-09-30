import torch

from .eru_base_workflow import EruBaseWorkflow

# ---------------------------------------------------------------------------

class EruBaseSimilarityWorkflow(EruBaseWorkflow):

    # -----------------------------------------------------------------------

    @classmethod
    def make_model(cls, example_stream, config):

        assert False, "Needs to be overridden"

    # -----------------------------------------------------------------------

    @classmethod
    def make_loss_fn(self):

        return torch.nn.MSELoss() # TODO consider hybrid with contrastive loss

    # -----------------------------------------------------------------------

    @classmethod
    def make_batch(cls, example_stream, batch_size, max_seq_len):

        utterances_1, utterances_2, labels = [], [], []
        for _k in range(batch_size):
            utterance_1, utterance_2, similarity = example_stream.get_example()
            utterance_1_normalized = cls.get_normalized_seq(utterance_1, max_seq_len=max_seq_len)
            utterance_2_normalized = cls.get_normalized_seq(utterance_2, max_seq_len=max_seq_len)
            utterances_1.append(torch.tensor(utterance_1_normalized))
            utterances_2.append(torch.tensor(utterance_2_normalized))
            labels.append(torch.tensor(similarity, dtype=torch.float))

        x_1 = torch.stack(utterances_1)
        x_2 = torch.stack(utterances_2)
        y = torch.stack(labels)
        return x_1, x_2, y
