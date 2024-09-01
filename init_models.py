import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import Transformer


def load_model(model, model_id, opt):
    path = f"experiment/{opt.core.experiment_id}/models/{model_id}/checkpoint.pt"
    model.load_state_dict(torch.load(path))


def init_embedding_weights_with_seed(model, strategy, seed):
    if strategy not in [
        "glorot_normal",
        "glorot_uniform",
        "kaiming_normal",
        "kaiming_uniform",
    ]:
        raise ValueError(
            f"Invalid strategy '{strategy}'. Please choose either 'gaussian' or 'glorot'."
        )

    torch.manual_seed(seed)
    if strategy == "glorot_normal":
        with torch.no_grad():
            nn.init.xavier_normal_(model.decoder.embed.embed.weight)
    elif strategy == "glorot_uniform":
        with torch.no_grad():
            nn.init.xavier_uniform_(model.decoder.embed.embed.weight)
    elif strategy == "kaiming_normal":
        with torch.no_grad():
            nn.init.kaiming_normal_(model.decoder.embed.embed.weight)
    elif strategy == "kaiming_uniform":
        with torch.no_grad():
            nn.init.kaiming_normal_(model.decoder.embed.embed.weight)


def get_base_model(opt):

    assert opt.model.d_model % opt.model.heads == 0
    assert opt.training1.dropout < 1

    torch.manual_seed(opt.core.base_init_seed)

    model = Transformer(
        opt.vocab_size,
        opt.model.d_model,
        opt.model.n_layers,
        opt.model.heads,
        opt.training1.dropout,
    )
    model.to(opt.device.device)
    model.eval()

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


def init_models(opt):
    model = get_base_model(opt)
    embed_init_seed = None
    embed_init_strategy = None
    if opt.model_id == 1:
        embed_init_strategy = opt.core.model1_embed_init
        embed_init_seed = opt.core.model1_embed_init_seed
    elif opt.model_id == 2:
        embed_init_strategy = opt.core.model2_embed_init
        embed_init_seed = opt.core.model2_embed_init_seed

    init_embedding_weights_with_seed(model, embed_init_strategy, seed=embed_init_seed)

    return model
