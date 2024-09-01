import logging
import math
import os
import random
import shutil
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from box import Box
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import GPT2TokenizerFast

import model

# from utils.options import Options
from configs import load_experiment_config
from data_utils import WikiDataset, create_masks, read_corpus
from init_models import init_models
from utils.color_print import cyan, green, orange, red

# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def seed_all(seed):
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    rng = np.random.default_rng(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return rng


seed_all(42)


def calculate_mse_torch(tensor1, tensor2):
    return torch.mean((tensor1 - tensor2) ** 2).item()


def save_loss(
    train_perplexities, valid_perplexities, test_perplexity, test_epoch, model_id, opt
):
    df = pd.DataFrame(
        {
            "train_perplexities": train_perplexities,
            "valid_perplexities": valid_perplexities,
        }
    )
    df.index.name = "Epoch"
    df["test_perplexities"] = None
    df.at[test_epoch, "test_perplexities"] = test_perplexity
    path = f"experiments/{opt.core.experiment_id}/models/{model_id}/perplexities.csv"
    df.to_csv(path)


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logging.info("Model and optimizer states have been loaded successfully.")


def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])


def save_checkpoint(model, optimizer, epoch, model_id, opt):
    path = f"experiments/{opt.core.experiment_id}/models/{model_id}/checkpoints/checkpoint_{epoch}.pt"
    os.makedirs(os.path.dirname(path))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
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
            logging.error(f"Failed to delete {file_path}. Reason: {e}")
    logging.info("Directory cleared")


def train_model(model, optimizer, opt, model_id):
    train_str = f"Training Model {model_id}"
    stars = "*" * len(train_str)
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

    for epoch in range(1, opt.training2.epochs + 1):
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

        avg_loss = total_loss / total_batches
        train_perplexity = math.exp(avg_loss)
        logging.info(f"Epoch: {epoch} - Train Perplexity: {train_perplexity}")
        train_perplexities.append(train_perplexity)

        valid_perplexity = test_model(model, opt, dataset="valid")
        logging.info(f"Epoch: {epoch} - Valid Perplexity: {valid_perplexity}")
        valid_perplexities.append(valid_perplexity)

        # save embeddings
        save_embeddings(model, model_id, epoch, opt)
        if (epoch) % 10 == 0 or epoch == 1:
            save_checkpoint(model, optimizer, epoch, model_id, opt)

    # Done training
    test_perplexity = test_model(model, opt, dataset="test")
    logging.info(f"Test Perplexity: {test_perplexity}")

    last_epoch = opt.training2.epochs
    plot_perplexity(train_perplexities, valid_perplexities, model_id, opt)
    save_loss(
        train_perplexities,
        valid_perplexities,
        test_perplexity,
        last_epoch,
        model_id,
        opt,
    )
    return model


def test_model(model, opt, dataset="valid"):
    model.eval()

    criterion = nn.CrossEntropyLoss()
    if dataset == "test":
        wiki_test_loader = opt.test_loader
    elif dataset == "valid":
        wiki_test_loader = opt.valid_loader
    elif dataset == "train":
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

    avg_loss = total_loss / total_batches
    return math.exp(avg_loss)


def plot_perplexity(train_perplexities, valid_perplexities, model_id, opt):
    plt.figure()
    plt.plot(train_perplexities, label="Train Perplexity")
    plt.plot(valid_perplexities, label="Valid Perplexity")
    if opt.experiment_core.plot_title:
        plt.title(opt.plot_title)
    else:
        plt.title(f"Model {model_id} Perplexity")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.legend()
    path = f"experiments/{opt.experiment_id}/models/{model_id}/Model {model_id} Perplexity.png"
    logging.info(f"Saving to {path}")
    plt.savefig(path)


def freeze_weights(model):
    for param in model.parameters():
        param.requires_grad = False
    model.decoder.embed.embed.weight.requires_grad = True


def dataloader_testing(opt):
    input_ids_run_path = (
        f"experiments/{opt.core.experiment_id}/models/{opt.model_id}/data/input_ids.txt"
    )
    create_folder_if_not_exists(os.path.dirname(input_ids_run_path))
    seed_all(42)
    with open(input_ids_run_path, "w") as f:
        for epoch in range(1, 3):
            for batch_idx, (input_ids, targets) in enumerate(tqdm(opt.train_loader)):
                f.write(f"Epoch {epoch}, Batch {batch_idx}:\n")
                f.write(" ".join(map(str, input_ids.tolist())) + "\n\n")

    return input_ids_run_path


# Source: https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097/7
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    print(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def experiment(opt):

    # opt = Options()
    # opt.make_vars(args_dict)
    # opt.device = 0 if opt.no_cuda is False else -1
    if not opt.device.no_cuda and torch.cuda.is_available():
        opt.torch_device = torch.device(opt.device.device)
    else:
        opt.torch_device = torch.device("cpu")

    exp_dir = f"experiments/{opt.core.experiment_id}"
    model_dir = os.path.join(exp_dir, "models", str(opt.model_id))
    if os.path.exists(model_dir):
        logging.info(
            f"Experiment {opt.core.experiment_id} model {opt.model_id} already exists"
        )
        logging.info("Do you wish to overwrite?")
        logging.info("y/n")
        response = input("Do you wish to overwrite? (y/n): ")
        if response.lower() == "y":
            logging.info("Overwriting")
            clear_directory(model_dir)
        else:
            logging.info("Exiting")
            sys.exit()
    else:
        os.makedirs(model_dir)
        os.makedirs(os.path.join(model_dir, "embeddings"))
        logging.info(f"Experiment directory created ({model_dir})")

    exp_str = (
        "=" * 10
        + f" Running Experiment {opt.core.experiment_id} Model {opt.model_id} "
        + "=" * 10
    )
    border = "=" * len(exp_str)
    green(border)
    green(exp_str)
    green(border)

    start_time = time.time()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    opt.vocab_size = tokenizer.vocab_size  # 50,257 from GPT2
    train_text = read_corpus(
        "data/wiki2.train.txt", tokenizer, first_n=opt.training2.train_subset
    )
    valid_text = read_corpus("data/wiki2.valid.txt", tokenizer)
    test_text = read_corpus("data/wiki2.test.txt", tokenizer)
    g = torch.Generator()
    wiki_train = WikiDataset(opt.model.seqlen, train_text, overlapping=True)
    g.manual_seed(0)
    wiki_train_loader = DataLoader(
        wiki_train,
        batch_size=opt.training1.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g,
    )

    wiki_valid = WikiDataset(opt.model.seqlen, valid_text, overlapping=True)
    wiki_valid_loader = DataLoader(
        wiki_valid,
        batch_size=opt.training1.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )

    wiki_test = WikiDataset(opt.model.seqlen, test_text, overlapping=True)
    wiki_test_loader = DataLoader(
        wiki_test,
        batch_size=opt.training1.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )

    opt.train_loader = wiki_train_loader

    opt.valid_loader = wiki_valid_loader

    opt.test_loader = wiki_test_loader

    model = init_models(opt)
    opt.optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.training1.lr, betas=(0.9, 0.98), eps=1e-9
    )

    save_checkpoint(
        model=model, optimizer=opt.optimizer, epoch=0, model_id=opt.model_id, opt=opt
    )
    if opt.core.lock_weights and opt.core.starter_model_path:
        load_model(model, opt.core.starter_model_path)
        # load_model(model2, opt.core.starter_model_path)
        freeze_weights(model)
        # freeze_weights(model2)
        # mse = calculate_mse_torch(
        #     model1.decoder.embed.embed.weight, model2.decoder.embed.embed.weight
        # )
        # assert mse == 0.0
        # TODO: make it so it can handle other init strategies in init_models.py
        # torch.manual_seed(1)
        # with torch.no_grad():
        #     nn.init.xavier_normal_(model1.decoder.embed.embed.weight)
        # torch.manual_seed(2)
        # with torch.no_grad():
        #     nn.init.xavier_normal_(model2.decoder.embed.embed.weight)
        # mse = calculate_mse_torch(
        #     model1.decoder.embed.embed.weight, model2.decoder.embed.embed.weight
        # )
        # logging.info(f"initial mse from preloaded and reinit: {mse}")

    # count parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    red(f"total params: {params}")
    logging.info(f"total params: {params}")

    # opt.optimizer2 = torch.optim.Adam(
    #     model2.parameters(), lr=opt.training1.lr, betas=(0.9, 0.98), eps=1e-9
    # )
    if opt.run.test_dataloader:
        dataloader_testing(opt)
    else:  # traing
        model = train_model(model, opt.optimizer, opt, model_id=opt.model_id)

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
    model_id = int(sys.argv[2])

    opt = load_experiment_config(f"exp{experiment_num}")
    opt.model_id = model_id
    print(opt.core.experiment_id)
    print(opt.model.d_model)

    create_folder_if_not_exists("experiments")
    experiment(opt)


if __name__ == "__main__":

    main()
