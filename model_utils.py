import os

import torch
import torch.nn as nn

from model import Transformer


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])


def save_checkpoint(model, optimizer, step, opt):
    path = os.path.join(opt.ckpt_dir, f"ckpt_{step}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def save_embeddings(model, step, opt):
    weights = model.decoder.embed.embed.weight.cpu().detach()
    path = os.path.join(opt.wte_dir, f"wte_step{step}.pt")
    torch.save(weights, path)


def freeze_non_wte_weights(model):
    for param in model.parameters():
        param.requires_grad = False
    model.decoder.embed.embed.weight.requires_grad = True


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
    model.to(opt.device)
    model.eval()

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


def init_wte(model, strategy, seed):
    if strategy not in [
        "glorot_normal",
        "glorot_uniform",
        "kaiming_normal",
        "kaiming_uniform",
    ]:
        raise ValueError(
            f"Invalid strategy '{strategy}'. Please choose from 'glorot_normal', 'glorot_uniform', 'kaiming_normal', or 'kaiming_uniform'."
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


def init_model(opt):
    model = get_base_model(opt)
    embed_init_strategy = opt.core[f"model{opt.model_id}_embed_init"]
    embed_init_seed = opt.core[f"model{opt.model_id}_embed_init_seed"]
    init_wte(model, embed_init_strategy, seed=embed_init_seed)

    return model
