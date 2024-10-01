import sys

import torch
import torch.nn as nn

import ds_utils
from configs import load_experiment_config
from model import Transformer


def load_checkpoint_model(model, path):
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])


@torch.no_grad()
def calculate_mse_between_models(model1, model2):
    mse_dict = {}

    for (name1, param1), (name2, param2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        if name1 != name2:
            print(f"Warning: Parameter names don't match. {name1} vs {name2}")
            continue

        if param1.shape != param2.shape:
            print(f"Warning: Parameter shapes don't match for {name1}. Skipping.")
            continue

        mse = torch.mean((param1 - param2) ** 2).item()
        mse_dict[name1] = mse

    return mse_dict


def main():

    if len(sys.argv) < 2:
        raise ValueError("specify config name")
    experiment_id = sys.argv[1]

    opt = load_experiment_config(f"{experiment_id}")

    ckpt = 0

    # model1_path = f"experiments/{opt.core.experiment_id}/models/1/ckpts/ckpt_{ckpt}.pt"
    # model2_path = f"experiments/{opt.core.experiment_id}/models/2/ckpts/ckpt_{ckpt}.pt"

    ckpt1 = 16000
    ckpt2 = 16000
    model1_path = f"experiments/1/models/1/ckpts/ckpt_{ckpt1}.pt"
    model2_path = f"experiments/1/models/2/ckpts/ckpt_{ckpt2}.pt"

    model1 = Transformer(
        opt.model.vocab_size,
        opt.model.d_model,
        opt.model.n_layers,
        opt.model.heads,
        opt.training1.dropout,
    )
    model2 = Transformer(
        opt.model.vocab_size,
        opt.model.d_model,
        opt.model.n_layers,
        opt.model.heads,
        opt.training1.dropout,
    )

    load_checkpoint_model(model1, model1_path)
    load_checkpoint_model(model2, model2_path)

    mse = ds_utils.mse(
        model1.decoder.embed.embed.weight, model2.decoder.embed.embed.weight
    )
    print("MSE between model 1 and model 2 embeddings", mse)

    mse_results = calculate_mse_between_models(model1, model2)

    for name, mse in mse_results.items():
        print(f"MSE for {name}: {mse}")


if __name__ == "__main__":
    main()
