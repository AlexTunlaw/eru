from .eru_language_site import EruLanguageSite
from .eru_language_builder import EruLanguageBuilder

def run_scripts():

    site = EruLanguageSite(data_dir="_data/_20240708_eru_language")

    builder = EruLanguageBuilder(site, name_key="20240708")

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
        # language stats
        #
        { "enabled": False,
            "step-method": EruLanguageBuilder.generate_language_data,
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
            "step-method": EruLanguageBuilder.train_gru_binary_classification,
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
        { "enabled": True,
            "step-method": EruLanguageBuilder.train_self_attention_binary_classification,
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
        { "enabled": True,
            "step-method": EruLanguageBuilder.train_self_attention_similarity,
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

