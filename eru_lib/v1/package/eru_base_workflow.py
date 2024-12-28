import torch

from fulcro_core import FulcroWindowBasedLossLevel

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

        assert False, "Needs to be overridden"

    # -----------------------------------------------------------------------

    @classmethod
    def make_loss_fn(cls):

        assert False, "Needs to be overridden"

    # -----------------------------------------------------------------------

    @classmethod
    def make_batch(cls, example_stream, batch_size, max_seq_len):

        assert False, "Needs to be overridden"

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

        loss_fn = cls.make_loss_fn()

        loss_level_detector, loss_threshold = cls.get_early_stop(config)

        loss_log = []
        #
        # training loop 
        #
        for i_batch in range(batch_count):

            model.zero_grad()

            # main part

            batch = cls.make_batch(example_stream, batch_size, max_seq_len)
            x = batch[:-1]
            y_labels = batch[-1]

            y_predicted = model.forward(
                *x,
                observe_fn=observe_forward_fn,
                i_batch=i_batch,
            )

            loss = loss_fn(y_predicted, y_labels)
            loss_value = loss.item()

            loss.backward()
            optimizer.step()

            # misc

            if (i_batch % log_every_n) == 0:
                print(f"batch: {i_batch}, loss: {loss_value:.05f}")

            if loss_level_detector is not None and loss_threshold is not None:
                loss_level_detector.observe(loss_value)
                if loss_level_detector.current_level <= loss_threshold:
                    break

            loss_log.append(loss_value)

            continue # training loop

        print(f"Final ({i_batch+1}) loss: {loss_value:05f}")
        
        model.eval() # IMPORTANT
        
        return {
            "model": model,
            "loss-log": loss_log
        }