import torch

from fulcro_core import FulcroWindowBasedLossLevel

# ---------------------------------------------------------------------------

class EruBaseWorkflow:
    
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
    def get_workflow_metric(cls, y_predicted: torch.Tensor, y_labels: torch.Tensor):
        return ""
    
    # -----------------------------------------------------------------------

    @classmethod
    def train(cls, example_stream, config, observe_fn):

        batch_size = config["batch-size"]
        batch_count = config["batch-count"]
        max_seq_len = config["max-seq-len"]
        log_every_n = config.get("log-every-n", 1)

        model = cls.make_model(example_stream, config)

        lr = config["optimizer"]["adam"]["lr"]
        if isinstance(lr, float):
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=lr,
                weight_decay=config["optimizer"].get("adam", {}).get("wd", 0.0)
            )
        else:
            optimizer = torch.optim.AdamW(lr(model))

        def accumulative_observe(ctx_cumulative, ctx, flush_accumulation=False):
            if observe_fn is not None:
                ctx_cumulative[str(ctx["kind"])] = ctx
                if flush_accumulation:
                    observe_fn(ctx_cumulative)
            return

        loss_fn = cls.make_loss_fn()

        loss_level_detector, loss_threshold = cls.get_early_stop(config)

        loss_log = []
        loss_value = None
        #
        # training loop 
        #
        for i_batch in range(batch_count):

            model.zero_grad()

            # main part

            batch = cls.make_batch(example_stream, batch_size, max_seq_len)
            x = batch[:-1]
            y_labels = batch[-1]

            ctx_cumulative = {
                "previous-loss": loss_value # (don't use loss itself to not create computation graph dependency on the previous iteration)
            }

            y_predicted = model.forward(
                *x,
                ctx=ctx_cumulative,
                observe_fn=lambda ctx: accumulative_observe(ctx_cumulative=ctx_cumulative, ctx=ctx)
            )

            loss = loss_fn(y_predicted, y_labels)
            accumulative_observe(ctx_cumulative=ctx_cumulative, ctx={
                "kind": "batch",
                "i-batch": i_batch,
                "x": x,
                "y-labels": y_labels,
                "y-predicted": y_predicted,
                "loss": loss
            }, flush_accumulation=True)

            loss_value = loss.item() / y_predicted.shape[0] * 100 # note: this loss normalization is purely to account for batches of various sizes

            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.layers[0].W_key, model.layers[1].W_key.grad.norm())
            # torch.nn.utils.clip_grad_norm_(model.layers[0].W_query, model.layers[1].W_query.grad.norm())
            # print(f"norms of grad of L0 key, query: {model.layers[0].W_key.grad.norm().item():.3f}, {model.layers[0].W_query.grad.norm().item():.3f}")
            # print(f"                 L1 key, query: {model.layers[1].W_key.grad.norm().item():.3f}, {model.layers[1].W_query.grad.norm().item():.3f}")
            # torch.nn.utils.clip_grad_norm_(model.layers[1].W_value, model.layers[0].W_value.grad.norm())
            # print(f"norms of grad of L0 value: {model.layers[0].W_value.grad.norm().item():.3f}")
            # print(f"                 L1 value: {model.layers[1].W_value.grad.norm().item():.3f}")

            optimizer.step()

            # misc

            if (i_batch % log_every_n) == 0:
                workflow_metrics_info = cls.get_workflow_metric(y_predicted, y_labels)
                if workflow_metrics_info:
                    workflow_metrics_info = ", " + workflow_metrics_info
                print(f"batch: {i_batch}, loss: {loss_value:.05f}{workflow_metrics_info}")

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