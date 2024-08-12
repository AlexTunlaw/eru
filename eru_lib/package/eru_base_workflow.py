import torch

from fero_core import FeroWindowBasedLossLevel

# ---------------------------------------------------------------------------

class EruBaseWorkflow:
    
    BOS, EOS, PAD, BASE = 0, 1, 2, 10

    # -----------------------------------------------------------------------

    @classmethod
    def get_normalized_seq(cls, utterance, max_seq_len):
        assert max_seq_len >= 3
        return \
            [cls.BOS] + \
            [cls.BASE + token for token in utterance[: max_seq_len - 2]] + \
            [cls.PAD] * max(max_seq_len - 2 - len(utterance), 0) + \
            [cls.EOS]

    # -----------------------------------------------------------------------

    @classmethod
    def get_early_stop(cls, config):
        early_stop = config.get("early-stop")
        if early_stop and early_stop != "None":
            parts = config["early-stop"].split("<=")
            assert len(parts) == 2
            loss_level_detector = eval(parts[0])
            min_loss_threshold = float(parts[1])
            return loss_level_detector, min_loss_threshold
        return None, None

    # -----------------------------------------------------------------------

    @classmethod
    def make_model(cls, example_stream, config):

        assert "Needs to be overridden"

    # -----------------------------------------------------------------------

    @classmethod
    def train(cls, example_stream, config):

        batch_size = config["batch-size"]
        batch_count = config["batch-count"]
        max_seq_len = config["max-seq-len"]
        log_every_n = config.get("log-every-n", 1)
        observe_forward_fn = config.get("observe-forward-fn")

        model = cls.make_model(example_stream, config)

        optimizer = torch.optim.AdamW(model.parameters(), # note it's actually AdamW
            lr=config["optimizer"]["adam"]["lr"],
            weight_decay=config["optimizer"].get("adam", {}).get("wd", 0.0)
        )

        loss_fn = torch.nn.BCELoss()

        loss_level_detector, loss_threshold = cls.get_early_stop(config)

        def make_batch():

            utterances, labels = [], []
            for _k in range(batch_size):
                utterance, label = example_stream.get_example()
                utterance_normalized = cls.get_normalized_seq(utterance, max_seq_len=max_seq_len)
                utterances.append(torch.tensor(utterance_normalized))
                labels.append(torch.tensor(label, dtype=torch.float))

            x = torch.stack(utterances)
            y = torch.stack(labels)
            return x, y

        #
        # training loop 
        #
        for i_batch in range(batch_count):

            model.zero_grad()

            # main part

            x, y_labels = make_batch()

            y_predicted = model.forward(
                x,
                observe_fn=observe_forward_fn,
                i_batch=i_batch,
            )

            loss = loss_fn(y_predicted, y_labels)

            loss.backward()
            optimizer.step()

            # misc

            if (i_batch % log_every_n) == 0:
                print(f"batch: {i_batch}, loss: {loss.item():.05f}")

            if loss_level_detector is not None and loss_threshold is not None:
                loss_level_detector.observe(loss.item())
                if loss_level_detector.current_level <= loss_threshold:
                    break

            continue # training loop

        print(f"Final ({i_batch+1}) loss: {loss.item():05f}")
        return