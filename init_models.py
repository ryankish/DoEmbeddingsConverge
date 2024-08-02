
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

from model import Transformer
import ds_utils

def load_model(model, model_id, opt):
    path = f"experiment/{opt.experiment_id}/models/{model_id}/checkpoint.pt"
    model.load_state_dict(torch.load(path))

def init_embedding_weights_with_seed(model, strategy, seed):
    if strategy not in ['glorot_normal', 'glorot_uniform', 'kaiming_normal','kaiming_uniform']:
        raise ValueError(f"Invalid strategy '{strategy}'. Please choose either 'gaussian' or 'glorot'.")

    torch.manual_seed(seed)
    if strategy=='glorot_normal':
        with torch.no_grad():
            nn.init.xavier_normal_(model.decoder.embed.embed.weight)
    elif strategy=='glorot_uniform':
        with torch.no_grad():
            nn.init.xavier_uniform_(model.decoder.embed.embed.weight)
    elif strategy=='kaiming_normal':
        with torch.no_grad():
            nn.init.kaiming_normal_(model.decoder.embed.embed.weight)
    elif strategy=='kaiming_uniform':
        with torch.no_grad():
            nn.init.kaiming_normal_(model.decoder.embed.embed.weight)

def get_base_model(opt, model_id=None):

    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    torch.manual_seed(opt.seed)

    model = Transformer(
        opt.vocab_size, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
    model.to(opt.device)
    model.eval()

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

def init_models(opt):
    model1 = get_base_model(opt)
    model2 = copy.deepcopy(model1)
    original_embed_weights = copy.deepcopy(model1.decoder.embed.embed.weight)

    with torch.no_grad():
        mse = ds_utils.mse(model1.decoder.embed.embed.weight, model2.decoder.embed.embed.weight)
        assert mse == 0.0

        init_embedding_weights_with_seed(model1, opt.model1_embed_init, seed=1)
        init_embedding_weights_with_seed(model2, opt.model2_embed_init, seed=2)

        mse = ds_utils.mse(model1.decoder.embed.embed.weight, original_embed_weights)
        print('MSE between original embedding and model 1 embedding', mse)

    return model1, model2
