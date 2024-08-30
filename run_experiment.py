import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from transformers import GPT2TokenizerFast

import math
import numpy as np
import pandas as pd

import time
import os
import shutil
import sys
from tqdm import tqdm
import importlib
from box import Box

import matplotlib.pyplot as plt
import seaborn as sns

import json
import logging

from data_utils import read_corpus, WikiDataset, create_masks
from init_models import init_models
from utils.color_print import green, red, cyan, orange
# from utils.options import Options
from configs import load_experiment_config

# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

g = torch.Generator()
g.manual_seed(0)

def calculate_mse_torch(tensor1, tensor2):
    return torch.mean((tensor1 - tensor2) ** 2).item()

def save_loss(train_perplexities, valid_perplexities, test_perplexity, test_epoch, model_id, opt):
    df = pd.DataFrame({'train_perplexities': train_perplexities, 'valid_perplexities': valid_perplexities})
    df.index.name = 'Epoch'
    df['test_perplexities'] = None
    df.at[test_epoch, 'test_perplexities'] = test_perplexity
    path = f"experiments/{opt.core.experiment_id}/models/{model_id}/perplexities.csv"
    df.to_csv(path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    logging.info("Model and optimizer states have been loaded successfully.")

def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

def save_checkpoint(model, optimizer, epoch, model_id, opt):
    path = f"experiments/{opt.core.experiment_id}/models/{model_id}/checkpoint_{epoch}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    logging.info(f"Checkpoint saved at epoch {epoch} for model {model_id}")


def save_embeddings(model, model_id, epoch, opt):
    weights = model.decoder.embed.embed.weight.cpu().detach()
    path = f"experiments/{opt.core.experiment_id}/models/{model_id}/embeddings/embed_weights_epoch_{epoch}.pt"
    torch.save(weights, path)


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logging.info(f"'{folder_path}' dir created.")

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file or link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove the directory and all its contents
        except Exception as e:
            logging.error(f'Failed to delete {file_path}. Reason: {e}')
    logging.info('Directory cleared')

def train_model(model, optimizer, opt, model_id):
    train_str = f"Training Model {model_id}"
    stars = '*' * len(train_str)
    cyan(stars)
    cyan(train_str)
    cyan(stars)

    sys.exit()

    save_embeddings(model, model_id, 0, opt)
    save_checkpoint(model, optimizer, 0, model_id, opt)

    criterion = nn.CrossEntropyLoss()
    train_loader = opt.train_loader

    train_perplexities = []
    valid_perplexities = []

    for epoch in range(1, opt.training2.epochs+1):
        model.train()
        total_loss = 0
        total_batches = 0
        logging.info(f"Epoch: {epoch} ... training")
        for input_ids, targets in tqdm(train_loader):
            input_ids = input_ids.to(opt.device)
            targets = targets.to(opt.device)
            input_mask = create_masks(input_ids)
            
            outputs = model(input_ids, input_mask)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            total_loss += loss.item()
            total_batches += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        avg_loss = total_loss/total_batches
        train_perplexity = math.exp(avg_loss)
        logging.info(f'Epoch: {epoch} - Train Perplexity: {train_perplexity}')
        train_perplexities.append(train_perplexity)

        valid_perplexity = test_model(model, opt, dataset='valid')
        logging.info(f'Epoch: {epoch} - Valid Perplexity: {valid_perplexity}')
        valid_perplexities.append(valid_perplexity)

        # save embeddings
        save_embeddings(model, model_id, epoch, opt)
        if (epoch) % 10 == 0 or epoch==1:
            save_checkpoint(model, optimizer, epoch, model_id, opt)


    # Done training
    test_perplexity = test_model(model, opt, dataset='test')
    logging.info(f'Test Perplexity: {test_perplexity}')

    last_epoch = opt.training2.epochs
    plot_perplexity(train_perplexities, valid_perplexities, model_id, opt)
    save_loss(train_perplexities, valid_perplexities, test_perplexity, last_epoch, model_id, opt)
    return model

def test_model(model, opt, dataset='valid'):
    model.eval()

    criterion = nn.CrossEntropyLoss()
    if dataset == 'test':
        wiki_test_loader = opt.test_loader
    elif dataset == 'valid':
        wiki_test_loader = opt.valid_loader
    elif dataset == 'train':
        wiki_test_loader = opt.train_loader
    total_loss = 0
    total_batches = 0

    with torch.no_grad():
        for input_ids, targets in tqdm(wiki_test_loader):
            input_ids = input_ids.to(opt.torch_device)
            targets = targets.to(opt.torch_device)
            input_mask = create_masks(input_ids)

            outputs = model(input_ids, input_mask)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss/total_batches
    return math.exp(avg_loss)

def plot_perplexity(train_perplexities, valid_perplexities, model_id, opt):
    plt.figure()
    plt.plot(train_perplexities, label='Train Perplexity')
    plt.plot(valid_perplexities, label='Valid Perplexity')
    if opt.experiment_core.plot_title:
        plt.title(opt.plot_title)
    else:
        plt.title(f'Model {model_id} Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    path = f'experiments/{opt.experiment_id}/models/{model_id}/Model {model_id} Perplexity.png'
    logging.info(f'Saving to {path}')
    plt.savefig(path)


def freeze_weights(model):
    for param in model.parameters():
        param.requires_grad = False
    model.decoder.embed.embed.weight.requires_grad = True

# def dataloader_testing(opt):
#
#     # run 1
#     for epoch in range(1, 3):
#         for input_ids, targets in tqdm(opt.train_loader):
#             # save the input_ids to a file to compare with run 2 later
#
#     # run 2
#     for epoch in range(1, 3):
#         for input_ids, targets in tqdm(opt.train_loader):
#             # save the input_ids to a file to compare with run 2 later
#
# def compare_input_ids_between_runs(input_ids_run1_path, input_ids_run2_path):
#     return
#

def experiment(opt):

    # opt = Options()
    # opt.make_vars(args_dict)
    # opt.device = 0 if opt.no_cuda is False else -1
    if not opt.device.no_cuda and torch.cuda.is_available():
        opt.torch_device = torch.device(opt.device.device)
    else:
        opt.torch_device = torch.device("cpu")
    
   
    directory = f"experiments/{opt.core.experiment_id}"
    if os.path.exists(directory):
        logging.info(f'Experiment {opt.core.experiment_id} already exists')
        logging.info('Do you wish to overwrite?')
        logging.info('y/n')
        response = input('Do you wish to overwrite? (y/n): ')
        if response.lower() == 'y':
            logging.info('Overwriting')
            clear_directory(directory)
        else:
            logging.info('Exiting')
            sys.exit()
    else:
        os.makedirs(directory)
        logging.info(f'Experiment directory created ({directory})')

    model1_dir = f'experiments/{opt.core.experiment_id}/models/1/embeddings'
    if not os.path.exists(model1_dir):
        os.makedirs(model1_dir)

    model2_dir = f'experiments/{opt.core.experiment_id}/models/2/embeddings'
    if not os.path.exists(model2_dir):
        os.makedirs(model2_dir)

    exp_str = f'='*10 + f' Running Experiment {opt.core.experiment_id} ' + '=' * 10
    border = '=' * len(exp_str)
    green(border)
    green( f'='*10 + f' Running Experiment {opt.core.experiment_id} ' + '=' * 10)
    green(border)


    start_time = time.time()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    opt.vocab_size = tokenizer.vocab_size # 50,257 from GPT2
    train_text = read_corpus('data/wiki2.train.txt',tokenizer, first_n=opt.training2.train_subset)
    valid_text = read_corpus('data/wiki2.valid.txt',tokenizer)
    test_text = read_corpus('data/wiki2.test.txt',tokenizer)

    wiki_train = WikiDataset(opt, train_text, overlapping=False)
    wiki_train_loader1 = DataLoader(wiki_train, batch_size=opt.training1.batch_size, seed=opt.data.dataset_seed, shuffle=True, drop_last=True, num_workers=2, worker_init_fn=seed_worker, generator=g)
    wiki_train_loader2 = DataLoader(wiki_train, batch_size=opt.training1.batch_size, seed=opt.data.dataset_seed, shuffle=True, drop_last=True, num_workers=2, worker_init_fn=seed_worker, generator=g)

    wiki_valid = WikiDataset(opt, valid_text, overlapping=False)
    wiki_valid_loader1 = DataLoader(wiki_valid, batch_size=opt.training1.batch_size, seed=opt.data.dataset_seed, shuffle=False, drop_last=True, num_workers=2, worker_init_fn=seed_worker, generator=g)
    wiki_valid_loader2 = DataLoader(wiki_valid, batch_size=opt.training1.batch_size, seed=opt.data.dataset_seed, shuffle=False, drop_last=True, num_workers=2, worker_init_fn=seed_worker, generator=g)


    wiki_test = WikiDataset(opt, test_test, overlapping=False)
    wiki_test_loader1 = DataLoader(wiki_test, batch_size=opt.training1.batch_size, seed=opt.data.dataset_seed, shuffle=False, drop_last=True, num_workers=2, worker_init_fn=seed_worker, generator=g)
    wiki_test_loader2 = DataLoader(wiki_test, batch_size=opt.training1.batch_size, seed=opt.data.dataset_seed, shuffle=False, drop_last=True, num_workers=2, worker_init_fn=seed_worker, generator=g)

    opt.train_loader1 = wiki_train_loader1
    opt.train_loader2 = wiki_train_loader2

    opt.valid_loader1 = wiki_valid_loader1
    opt.valid_loader2 = wiki_valid_loader2

    opt.test_loader1 = wiki_test_loader1
    opt.test_loader2 = wiki_test_loader2


    model1, model2 = init_models(opt)
    if opt.core.lock_weights and opt.core.starter_model_path:
        load_model(model1, opt.core.starter_model_path)
        load_model(model2, opt.core.starter_model_path)
        freeze_weights(model1)
        freeze_weights(model2)
        mse = calculate_mse_torch(model1.decoder.embed.embed.weight, model2.decoder.embed.embed.weight)
        assert mse == 0.0
        # TODO: make it so it can handle other init strategies in init_models.py
        torch.manual_seed(1)
        with torch.no_grad():
            nn.init.xavier_normal_(model1.decoder.embed.embed.weight)
        torch.manual_seed(2)
        with torch.no_grad():
            nn.init.xavier_normal_(model2.decoder.embed.embed.weight)
        mse = calculate_mse_torch(model1.decoder.embed.embed.weight, model2.decoder.embed.embed.weight)
        logging.info(f'initial mse from preloaded and reinit: {mse}')
    

    # count parameters
    model_parameters = filter(lambda p: p.requires_grad, model1.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    red(f'total params: {params}')
    logging.info(f'total params: {params}')

    opt.optimizer1 = torch.optim.Adam(model1.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    opt.optimizer2 = torch.optim.Adam(model2.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)

    #
    # model1 = train_model(model1, opt.optimizer1, opt, model_id=1)
    # if opt.core.experiment_id != 0:
    #     model2 = train_model(model2, opt.optimizer2, opt, model_id=2)
    #
    
    green(f"Time taken: {time.time() - start_time}")

def main():

    if len(sys.argv) < 2:
        raise ValueError("specify config name")
    experiment_num = sys.argv[1]

    opt = load_experiment_config(f"exp{experiment_num}")
    print(opt.core.experiment_id)
    print(opt.model.d_model)

    opt.test_val = "test this calue"
    print(opt.test_val)
    # config_name = sys.argv[1]
    # opt = Box(flatten_config(config_name))
    # print(flat_config.n_layers)  # This will print 6 
    #
    create_folder_if_not_exists('experiments')
    experiment(opt)

    # Experiment 0
    # experiment_id = 0
    # experiment0_embedding_size = 512
    # model1_embed_init = 'glorot_uniform'
    # model2_embed_init = 'glorot_uniform'
    #
    # args0_dict = {
    #     'experiment_id': experiment_id,
    #     'seed': 0,
    #     'no_cuda': False,
    #     'SGDR': False,
    #     'epochs': 40,
    #     'model1_embed_init': model1_embed_init,
    #     'model2_embed_init': model2_embed_init,
    #     'd_model': experiment0_embedding_size,
    #     'n_layers': 6,
    #     'heads': 8,
    #     'dropout': 0.1,
    #     'batch_size': 3,
    #     'printevery': 100,
    #     'lr': 0.00001,
    #     'seqlen': 512,
    #     'threshold': 3,
    #     'norm': 2.0,
    #     'verbose': False,
    #     'device': 0,
    #     'time_name': None,
    #     'train_subset': None, # for testing purposes only
    #     'train': None,
    #     'valid': None,
    #     'test': None,
    #     'optimizer': None,
    #     'sched': None,
    #     'plot_title': 'Baseline Model Perplexity',
    # }
    #
    # experiment(args0_dict)
    #
    #
    # experiment_id = 2
    # experiment2_embedding_size = 128
    # model1_embed_init = 'glorot_uniform'
    # model2_embed_init = 'glorot_uniform'
    #
    # args_dict = {
    #     'experiment_id': experiment_id,
    #     'dataset_seed': 0,
    #     'device': "cuda:0",
    #     'no_cuda': False,
    #     'SGDR': False,
    #     'epochs': 2, # TODO: plot freezes
    #     'model1_embed_init': model1_embed_init,
    #     'model2_embed_init': model2_embed_init,
    #     'd_model': experiment2_embedding_size,
    #     'n_layers': 6,
    #     'heads': 8,
    #     'dropout': 0.1,
    #     'batch_size': 3,
    #     'printevery': 1, # TODO: implement
    #     'lr': 0.00001,
    #     'seqlen': 512,
    #     'threshold': 3,
    #     'norm': 2.0,
    #     'verbose': False,
    #     'time_name': None,
    #     'train_subset': None, # for testing purposes only
    #     'train': None,
    #     'valid': None,
    #     'test': None,
    #     'optimizer': None,
    #     'sched': None,
    #     'plot_title': None,
    #     'lock_weights': True,
    #     'starter_model_path': 'experiments/1/models/1/checkpoint_40.pt',
    # }
    #
    # log_filename = f'experiments/{experiment_id}/experiment_{experiment_id}.log'
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format='%(asctime)s - %(levelname)s - %(message)s'
    # )
    # file_handler = logging.FileHandler(log_filename)
    # logging.getLogger().addHandler(file_handler)
    #
    #
    # logging.info('Experiment arguments: %s', json.dumps(args_dict, indent=4))
    #
    # experiment(args_dict)
    #
if __name__ == '__main__':

    main()
    # Experiment 1
    # experiment_id = 1
    # experiment1_embedding_size = 128
    # model1_embed_init = 'glorot_uniform'
    # model2_embed_init = 'glorot_uniform'

    # args1_dict = {
    #     'experiment_id': experiment_id,
    #     'seed': 0,
    #     'no_cuda': False,
    #     'SGDR': False,
    #     'epochs': 40,
    #     'model1_embed_init': model1_embed_init,
    #     'model2_embed_init': model2_embed_init,
    #     'd_model': experiment1_embedding_size,
    #     'n_layers': 6,
    #     'heads': 8,
    #     'dropout': 0.1,
    #     'batch_size': 3,
    #     'printevery': 100,
    #     'lr': 0.00001,
    #     'seqlen': 512,
    #     'threshold': 3,
    #     'norm': 2.0,
    #     'verbose': False,
    #     'device': 0,
    #     'time_name': None,
    #     'train_subset': None, # for testing purposes only
    #     'train': None,
    #     'valid': None,
    #     'test': None,
    #     'optimizer': None,
    #     'sched': None,
    #     'plot_title': None,
    # }

    # experiment(args1_dict)

    # =========== to recreate the perplexity plot
    # model_id = 1
    # opt = Options()
    # opt.make_vars(args1_dict)
    # exp_perplexities_model = pd.read_csv(f'experiments/{opt.experiment_id}/models/{model_id}/perplexities.csv')
    # train_perplexities, valid_perplexities = exp_perplexities_model['train_perplexities'].values, exp_perplexities_model['valid_perplexities'].values
    # plot_perplexity(train_perplexities, valid_perplexities, model_id, opt)

    # Experiment 2
    
