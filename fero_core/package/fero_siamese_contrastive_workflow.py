import sys
import numpy as np
import logging

import torch
mps_torch_device = torch.device("mps") if torch.backends.mps.is_available() else None

from ml3 import ConfigChecker, SimpleObject

from .fero_siamese_learner import FeroSiameseLearner
from .fero_contrastive_loss import FeroContrastiveLoss
from .fero_window_based_loss_level import FeroWindowBasedLossLevel

class FeroSiameseContrastiveWorkflow:

    # -----------------------------------------------------------------------

    @classmethod
    def make_batch(cls, stream, example_count):
        features = list(stream.get_examples(example_count))
        np.random.shuffle(features) # (important so that positives in batches keep getting matched up with different negatives)
        return torch.stack(features)

    # -----------------------------------------------------------------------

    @classmethod
    def make_centroids(cls, model, positive_training_stream, centroid_count, examples_per_centroid, torch_device=None):

        centroids = []
        for i_centroid in range(centroid_count):
            batch = cls.make_batch(positive_training_stream, examples_per_centroid)
            if torch_device is not None:
                batch = batch.to(torch_device)
            projected = model.forward_1(batch)
            centroid = torch.mean(projected, dim=0)
            centroids.append(centroid)

        return centroids

    # -----------------------------------------------------------------------

    @classmethod
    def get_early_stop(cls, cfg):
        early_stop = cfg.get("early-stop")
        if early_stop and early_stop != "None":
            parts = cfg["early-stop"].split("<=")
            assert len(parts) == 2
            loss_level_detector = eval(parts[0])
            min_loss_threshold = float(parts[1])
            return loss_level_detector, min_loss_threshold
        return None, None

    # -----------------------------------------------------------------------

    @classmethod
    def train_model(cls,
        training_config,
        positive_training_stream,
        negative_training_stream,
        use_mps_device=False,
        log_as_print=False,
    ):

        log = print if log_as_print else logging.info

        cfg = ConfigChecker.check_paths_in_config(training_config, [
            "siamese/input-dim", "siamese/contrastive-head-dim",
            "centroids/count", "centroids/examples-per-centroid",
            "optimizer", 
            "batch/count", "batch/size",
        ])

        if mps_torch_device is None:
            use_mps_device = False
        torch_device = mps_torch_device if use_mps_device else torch.device("cpu")

        model = FeroSiameseLearner(
            input_dim=cfg["siamese"]["input-dim"],
            contrastive_head_dim=cfg["siamese"]["contrastive-head-dim"],
            dtype=cfg["siamese"].get("dtype", torch.float32),
            dropout=cfg["siamese"].get("dropout")
        ).to(torch_device)

        optimizer = torch.optim.Adam(model.parameters(),
            lr=cfg["optimizer"]["adam"]["lr"],
            weight_decay=cfg["optimizer"].get("adam", {}).get("wd", 0.0)
        )

        sim_fn = torch.nn.CosineSimilarity(dim=1)
        loss_fn = FeroContrastiveLoss()

        batch_size=cfg["batch"]["size"]

        loss_level_detector, loss_threshold = cls.get_early_stop(cfg)

        #
        # training loop 
        #

        for i_batch in range(cfg["batch"]["count"]):
            model.zero_grad()

            # batch construction:
            # first half: pairs positive - positive; and label = 1.0
            # second half: pairs positive - negative; and label = 0.0
            x_1 = cls.make_batch(positive_training_stream, batch_size).to(torch_device)
            x_2 = torch.cat([
                cls.make_batch(positive_training_stream, int(batch_size // 2)),
                cls.make_batch(negative_training_stream, int(batch_size // 2))
            ]).to(torch_device)

            label = torch.FloatTensor(([1.] * int(batch_size//2)) + ([0.] * int(batch_size//2))).to(torch_device)

            y_1, y_2 = model.forward(x_1, x_2)
            sim = sim_fn(y_1, y_2)
            dist = 1. - sim
            loss = loss_fn(dist, label)

            loss.backward()
            optimizer.step()

            # misc
            if (i_batch % 2) == 0:
                log(f"batch: {i_batch}, loss: {loss.item():.05f}")

            if loss_level_detector is not None and loss_threshold is not None:
                loss_level_detector.observe(loss.item())
                if loss_level_detector.current_level <= loss_threshold:
                    break

            continue # training loop
                
        log(f"Final ({i_batch+1}) loss: {loss.item():05f}")

        centroids = cls.make_centroids(model,
            positive_training_stream=positive_training_stream, 
            centroid_count=cfg["centroids"]["count"],
            examples_per_centroid=cfg["centroids"]["examples-per-centroid"],
            torch_device=torch_device
        )

        if "saved-model" in cfg["output"]:
            model.save(cfg["output"]["saved-model"], centroids=centroids)
        else:
            model.centroids = centroids

        model.eval() # important (e.g. to disable the dropout)
        return SimpleObject(model=model)
