from nit import NiT
from minit import MINiT

# default hyperparameters
def default():
    hyperparameters = {
        "gpus": 8,
        "dim": 256,
        "heads": 8,
        "depth": 6,
        "mlp_dim": 256,
        "dropout": 0.3,
        "emb_dropout": 0.4,
        "weight_decay": 0.01,
        "drophead": 0.25,
        "layerdrop": 0.25,
        "cutmix": 0.7,
        "mixup": 0.3,
        "alpha": 1,
        "multiplier": 1,
        "warmup_epochs": 2,
        "scaling_epochs": 6,
        "lr": 1e-4,
        "image_size": 64,
        "patch_size": 8,
        "num_classes": 2,
        "channels": 1,
        "block_size": 16,
        "patch_size": 4,
        "load_pretrained": False,
        "load_moco": False,
        "distillation": False,
        "amp_enabled": True,
        "augment": True,
        "finetune": False,
        "datasets": ["abcd", "ncanda"],
    }
    hyperparameters["batch_size"] = 32 * hyperparameters["gpus"]
    hyperparameters["dim_head"] = hyperparameters["dim"] // hyperparameters["heads"]
    return hyperparameters

def nit_hyperparameters():
    hyperparameters = default()
    hyperparameters["model"] = NiT
    hyperparameters["dimension"] = 128
    hyperparameters["cutmix"] = 0.51
    hyperparameters["mixup"] = 0.345
    hyperparameters["heads"] = 8
    hyperparameters["dropout"] = 0.315
    hyperparameters["emb_dropout"] = 0.0862
    hyperparameters["weight_decay"] = 0.165
    hyperparameters["warmup_epochs"] = 27
    hyperparameters["depth"] = 4
    hyperparameters["mlp_dim"] = 234
    hyperparameters["lr"] = 1e-4
    hyperparameters["alpha"] = 0.62
    hyperparameters["multiplier"] = 1.313
    hyperparameters["scaling_epochs"] = 4
    hyperparameters["patch_size"] = 8
    hyperparameters["batch_size"] = 40 * hyperparameters["gpus"]
    return hyperparameters


def minit_hyperparameters():
    hyperparameters = default()
    hyperparameters["model"] = MINiT
    hyperparameters["dimension"] = 512
    hyperparameters["cutmix"] = 0.36
    hyperparameters["mixup"] = 0.149
    hyperparameters["heads"] = 8
    hyperparameters["dropout"] = 0.07
    hyperparameters["emb_dropout"] = 0.016
    hyperparameters["weight_decay"] = 0.125
    hyperparameters["warmup_epochs"] = 16
    hyperparameters["depth"] = 6
    hyperparameters["mlp_dim"] = 309  # 256?
    hyperparameters["lr"] = 1e-4
    hyperparameters["alpha"] = 0.98
    hyperparameters["multiplier"] = 1.16
    hyperparameters["scaling_epochs"] = 42
    hyperparameters["distillation"] = True
    hyperparameters["batch_size"] = 24 * hyperparameters["gpus"]
    return hyperparameters

