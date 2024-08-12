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
        # training data example
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
        { "enabled": False,
            "step-method": EruLanguageBuilder.train_gru,
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
        { "enabled": True,
            "step-method": EruLanguageBuilder.train_self_attention,
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
                    # "attention-weights-mode": "softmax",
                    # "attention-weights-mode": "sigmoid",
                    "attention-weights-mode": "overtaking-sigmoid",
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

