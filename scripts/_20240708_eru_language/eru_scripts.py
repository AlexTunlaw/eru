from .eru_site import EruSite
from .eru_builder_e1 import EruBuilderE1
from .eru_builder_e2 import EruBuilderE2, AttentionLevelsObserver

# ---------------------------------------------------------------------------

def run_scripts():

    {
        "e1": run_scripts_e1,
        "e2": run_scripts_e2,
    }[
        "e2"
    ]       ()

    return

# ---------------------------------------------------------------------------

def run_scripts_e1():

    site = EruSite(data_dir="_data/_20240708_eru_language")

    builder = EruBuilderE1(site, name_key="20240708")

    language_params = {
        "vocab-size": 2000,
        "build-steps": 2000,
        "p-vocab-expansion": 0.2,
        "p-modifier-expansion": 0.5,
        "p-modifier-reuse": 0.1,
        "max-expression-depth": 5,
        "utterance-min-len": 3,
        "utterance-max-len": 10,
    }

    builder.run_steps([
        #
        # language data
        #
        { "enabled": False,
            "step-method": EruBuilderE1.generate_e1_language_data,
            "input": "__NONE__",
            "outputs": {
                "training-data": "training-data: csv",
            },
            "training-data-columns": (
                "utterance",
                "signature"
            ),
            **language_params,
            "training-data-count": 300,
        },
        #
        # binary classification experiments - GRU; baseline to show that the language is learnable
        #
        { "enabled": False,
            "step-method": EruBuilderE1.train_e1_gru_binary_classification,
            "language": language_params,
            "training-config": {
                "early-stop": "FeroWindowBasedLossLevel() <= 0.05", # note
                "batch-size": 100,
                "batch-count": 250,
                "max-seq-len": 12,
                "log-every-n": 10,
                "model": {
                    "embedding-dim": 64,
                    "gru-hidden-dim": 50
                },
                "optimizer": {
                    "adam": {
                        "lr": 0.01,
                        "wd": 0.05
                    }
                }
            }
        },
        #
        # binary classification experiments - Multi-head Self-Attention
        #
        { "enabled": False,
            "step-method": EruBuilderE1.train_e1_self_attention_binary_classification,
            "outputs": {
                "loss-logs": "loss-logs: csv",
            },
            "run-count": 5,
            "language": language_params,
            "training-config": {
                "early-stop": "FeroWindowBasedLossLevel() <= 0.05", # note
                "batch-size": 100,
                "batch-count": 250,
                "max-seq-len": 12,
                "log-every-n": 10,
                "model": {
                    #
                    # main results here
                    #
                    "attention-weights-mode": "softmax", # 126.4 steps to diverge
                    # "attention-weights-mode": "sigmoid",  # 77.0 mean steps to converge (some temporary instability/divergence)
                    # "attention-weights-mode": "overtaking-sigmoid", # 64.0 mean steps to converge; 1/5 runs diverges
                    # "attention-weights-mode": "overtaking-sigmoid-tuned", # 64.4 mean steps to converge
                    "embedding-dim": 64,
                    "attention-dim": 50,
                    "c-heads": 3
                },
                "optimizer": {
                    "adam": {
                        "lr": 0.01,
                        "wd": 0.05
                    }
                }
            }
        },
        #
        # similarity experiments - Multi-head Self-Attention
        #
        { "enabled": False,
            "step-method": EruBuilderE1.train_e1_self_attention_similarity,
            "outputs": {},
            "run-count": 5,
            "language": language_params,
            "example-stream": {
                "p-basically-similar": 0.5,
                "similarity-if-basically-similar": 0.66,
                "expression-vocab-loss": 0.1,
                "modifiers-weight": 0.4
            },
            "training-config": {
                "early-stop": "FeroWindowBasedLossLevel() <= 0.05", # note
                "batch-size": 100,
                "batch-count": 250,
                "max-seq-len": 12,
                "log-every-n": 10,
                "model": {
                    #
                    # main results here
                    #
                    # "attention-weights-mode": "softmax", # 135.0 mean steps to converge
                    # "attention-weights-mode": "sigmoid", # 81.4 steps to converge
                    "attention-weights-mode": "overtaking-sigmoid", # 73.0 steps to converge ** winner
                    # "attention-weights-mode": "overtaking-sigmoid-tuned", # 84.6 mean steps to converge
                    "embedding-dim": 64,
                    "attention-dim": 50,
                    "c-heads": 3
                },
                "optimizer": {
                    "adam": {
                        "lr": 0.01,
                        "wd": 0.05
                    }
                }
            }
        }
    ])

    print("DONE")

# ---------------------------------------------------------------------------

def run_scripts_e2():

    site = EruSite(data_dir="_data/_20240930_eru_language")

    builder = EruBuilderE2(site, name_key="20240930")

    utterance_len = 20
    language_params = {
        "language": {
            "vocab-size": 100, # 0 is EOU (End Of Utterance)
            "vocab-ps": [0.0, 0.2, 0.2] + ([0.6 / 97] * 97),
            "classes": {
                0: (0.5, "*"),
                1: (0.5, [1, 2])
            },
            "utterance-len": utterance_len,
        }
    }

    builder.run_steps([
        #
        # language data
        #
        { "enabled": False,
            "step-method": EruBuilderE2.generate_e2_language_data,
            "input": "__NONE__",
            "output": "training-data: csv",
            "output-columns": (
                "utterance",
                "signature"
            ),
            "output-count": 300,
            **language_params,
        },
        #
        # binary classification experiments - GRU; baseline to show that the language is learnable
        #
        { "enabled": False,
            "step-method": EruBuilderE2.train_e2_gru_binary_classification,
            **language_params,
            "training-config": {
                "early-stop": "FeroWindowBasedLossLevel() <= 0.05", # note
                "batch-size": 100,
                "batch-count": 250,
                "max-seq-len": utterance_len,
                "log-every-n": 10,
                "model": {
                    "embedding-dim": 64,
                    "gru-hidden-dim": 32
                },
                "optimizer": {
                    "adam": {
                        "lr": 0.01,
                        "wd": 0.05
                    }
                }
            }
        },
        #
        # binary classification experiments - Multi-head Self-Attention
        #
        { "enabled": False,
            "step-method": EruBuilderE2.train_e2_self_attention_binary_classification,
            "outputs": {},
            "run-count": 5,
            **language_params,
            "training-config": {
                "early-stop": "FeroWindowBasedLossLevel() <= 0.10", # note
                "batch-size": 100,
                "batch-count": 500,
                "max-seq-len": utterance_len,
                "log-every-n": 10,
                "model": {
                    "embedding-dim": 16,
                    "attention-dim": 14,
                    "c-heads": 1, # FIXED, strictly 1 head
                    # MAIN OBSERVATION here
                    # "c-layers": 1, # 1 not supposed to converge (and doesn't, with loss flat at 0.4)
                    "c-layers": 2, # 2: converges at 64.6 steps (of 5 runs) [It's interesting that this is faster than 3 heads]
                },
                "optimizer": {
                    "adam": {
                        "lr": 0.02,
                        "wd": 0.01
                    }
                }
            }
        },
        { "enabled": False,
            "step-method": EruBuilderE2.train_e2_self_attention_binary_classification,
            "outputs": {},
            "run-count": 5,
            **language_params,
            "training-config": {
                "early-stop": "FeroWindowBasedLossLevel() <= 0.10", # note
                "batch-size": 100,
                "batch-count": 500,
                "max-seq-len": utterance_len,
                "log-every-n": 10,
                "model": {
                    "embedding-dim": 16,
                    "attention-dim": 14,
                    "c-heads": 3, # FIXED at 3, allows for head interaction learning
                    # MAIN OBSERVATION here
                    # "c-layers": 1, # 1: converges via head interaction at 70.4 (of 5 runs)
                    "c-layers": 2, # 2: converges at 70.8 steps (of 5 runs). ?: Via head interaction or layers?
                },
                "optimizer": {
                    "adam": {
                        "lr": 0.02,
                        "wd": 0.01
                    }
                }
            }
        },
        #
        # 10/26/2024 experimentation
        #
        # def get_weights_momentum(x):
        #     return x + (1 - x) * x**2.0
        # self.last_attention_weights = attention_weights
        # self.last_intrinsic_attention_weights_momentum = get_weights_momentum(attention_weights)
        # intrinsic_loss_fn = torch.nn.MSELoss()
        # intrinsic_loss_fn(
        #     model.layers[-1].last_attention_weights,
        #     model.layers[-1].last_intrinsic_attention_weights_momentum
        # )
        #
        # 2 transformer layers, momentum: x is attention_weights, x + (1 - x) * x**2.0 (gamma is 2.0); lr 0.02, wd 0.01
        # No intrinsic pretraining: 44.6 <-- baseline
        # with 50 steps of intrinsic pretraining: 88.8 - 50 = 38.8
        # with 100 steps of intrinsic pretraining: 135.8 - 100 = 35.8 <-- winner, but at a price of +100 pretraining steps
        # with every other step of intrinsic pretraining: 61.6
        # with 1, 2 out of 10 steps of intrinsic pretraining: 52.8
        # with sum-mix 1.0, 100.0 with intrinsic pretraining: 40.0
        # with sum-mix 1.0, 10.0 with intrinsic pretraining: 36.4 <-- winner
        #
        # 2 transformer layers, momentum: x is attention_weights, x + (1 - x) * x**2.0 (gamma is 2.0); lr 0.05, wd 0.0
        # No intrinsic pretraining: 20.2 <-- baseline
        # with 10 steps of intrinsic pretraining: 33.8 - 10 = 22.8
        #
        { "enabled": False,
            "step-method": EruBuilderE2.train_e2_self_attention_binary_classification,
            "outputs": {},
            "run-count": 5,
            **language_params,
            "training-config": {
                "early-stop": "FeroWindowBasedLossLevel() <= 0.15", # note
                "batch-size": 100,
                "batch-count": 500,
                "max-seq-len": utterance_len,
                "log-every-n": 10,
                "model": {
                    "embedding-dim": 16,
                    "attention-dim": 14,
                    "c-heads": 3,
                    # "c-layers": 1,
                    "c-layers": 2,
                },
                "optimizer": {
                    "adam": {
                        "lr": 0.05, # 0.02,
                        "wd": 0.0, # 0.01,
                    }
                }
            }
        },
        #
        # baseline 10/27/2024
        #
        # steps to converge: 20.6 of 5 runs
        #
        { "enabled": False,
            "step-method": EruBuilderE2.train_e2_self_attention_binary_classification,
            "outputs": {},
            "run-count": 5,
            **language_params,
            "training-config": {
                "early-stop": "FeroWindowBasedLossLevel() <= 0.15", # note
                "batch-size": 100,
                "batch-count": 200,
                "max-seq-len": utterance_len,
                "log-every-n": 10,
                "model": {
                    "embedding-dim": 16,
                    "attention-dim": 14,
                    "c-heads": 1, #NOTE 1: easy to see convergence patterns; 3: fast, robust convergence
                    # "c-layers": 1,
                    "c-layers": 2,
                },
                "optimizer": {
                    "adam": {
                        "lr": 0.01, # 0.02,
                        "wd": 0.0, # 0.01,
                    }
                }
            },
            "observation-mode": "layers 0,1 - 01, 10, 02", # "layers 0,1 - 01, 10, 02", "layer 0 - EOS to 0, 1, 2"
        },
        #
        # individual learning rates 11/16/2024
        #
        # Uniform, 0.01: converges in 14.2 steps, of 15 runs
        #
        { "enabled": True,
            "step-method": EruBuilderE2.train_e2_self_attention_binary_classification,
            "outputs": {},
            "run-count": 5,
            **language_params,
            "training-config": {
                "early-stop": "FeroWindowBasedLossLevel() <= 0.25", # note
                "batch-size": 100,
                "batch-count": 200,
                "max-seq-len": utterance_len,
                "log-every-n": 10,
                "model": {
                    "embedding-dim": 16,
                    "attention-dim": 14,
                    "c-heads": 1, #NOTE 1: easy to see convergence patterns; 3: fast, robust convergence
                    # "c-layers": 1,
                    "c-layers": 2,
                },
                "optimizer": {
                    "adam": {
                        "lr": lambda model: [
                            {"params": model.embedding.parameters(), "lr": 0.01 / 4},
                            {"params": model.layers[0].parameters(), "lr": 0.01 / 2},
                            {"params": model.layers[1].parameters(), "lr": 0.01},
                            {"params": model.fc.parameters(), "lr": 0.01 * 2},
                        ],
                        "wd": 0.0, # 0.01,
                    }
                }
            },
            "make-observer-fn": lambda: AttentionLevelsObserver(
                # observation_target="layer 0 - EOS to 0, 1, 2",
                observation_target="layers 0,1 - 01, 10, 02, EOS",
            ),
        },
    ])

    print("DONE")
