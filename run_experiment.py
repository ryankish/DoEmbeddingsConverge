import logging
import math
import os
import random
import shutil
import sys
import time
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from box import Box
from colorama import Back, Fore, Style, init
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from configs import load_experiment_config
from data_utils import WikiDataset, create_masks, read_corpus
from model_utils import (
    freeze_non_wte_weights,
    init_model,
    init_wte,
    load_checkpoint,
    load_model,
    save_checkpoint,
    save_embeddings,
)

# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def seed_all(seed):
    if not seed:
        seed = 10

    logging.info("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    rng = np.random.default_rng(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return rng


init(autoreset=True)


def setup_logging(opt):
    # Remove all handlers associated with the root logger object
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    class ColorFormatter(logging.Formatter):
        FADE = "\033[2m"  # Dim text
        RESET = "\033[0m"  # Reset all attributes
        FORMAT = f"{FADE}%(asctime)s - %(levelname)s -{RESET} %(message)s"
        FORMATS = {
            logging.DEBUG: Fore.CYAN + FORMAT + Style.RESET_ALL,
            logging.INFO: Fore.GREEN + FORMAT + Style.RESET_ALL,
            logging.WARNING: Fore.YELLOW + FORMAT + Style.RESET_ALL,
            logging.ERROR: Fore.RED + FORMAT + Style.RESET_ALL,
            logging.CRITICAL: Back.RED + Fore.WHITE + FORMAT + Style.RESET_ALL,
        }

        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)

    file_handler = logging.FileHandler(opt.log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(ColorFormatter())

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Disable propagation to the root logger
    logger.propagate = False


def log_cfg(cfg, indent=0):
    for key, value in cfg.items():
        if isinstance(value, Box):
            logging.info(" " * indent + f"{key}:")
            log_cfg(value, indent + 4)
        else:
            logging.info(" " * indent + f"{key}: {value}")


def calculate_mse_torch(tensor1, tensor2):
    return torch.mean((tensor1 - tensor2) ** 2).item()


def save_loss(
    train_perplexities, valid_perplexities, test_perplexity, current_step, opt
):
    df = pd.DataFrame(
        {
            "train_perplexities": train_perplexities,
            "valid_perplexities": valid_perplexities,
        },
        index=range(
            opt.training2.evaluate_every_n_steps,
            current_step + 1,
            opt.training2.evaluate_every_n_steps,
        ),
    )
    df.index.name = "Step"
    df["test_perplexities"] = None
    df.at[current_step, "test_perplexities"] = test_perplexity
    path = os.path.join(opt.out_dir, "perplexities.csv")
    df.to_csv(path)


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logging.info(f"'{folder_path}' dir created.")


def rm_dir(directory_path):
    try:
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
            logging.info(
                f"Directory '{directory_path}' and its contents have been removed successfully."
            )
        else:
            logging.info(f"Directory '{directory_path}' does not exist.")
    except Exception as e:
        logging.info(f"An error occurred while removing the directory: {e}")


def train_model(model, optimizer, opt, model_id):
    train_str = f"Training Model {model_id}"
    stars = "*" * len(train_str)
    logging.info(stars)
    logging.info(train_str)
    logging.info(stars)
    save_embeddings(model, 0, opt)
    save_checkpoint(model, optimizer, 0, opt)
    criterion = nn.CrossEntropyLoss()
    train_loader = opt.train_loader
    train_perplexities = []
    valid_perplexities = []

    total_steps = opt.training2.total_steps
    current_step = 0

    model.train()
    if opt.training2.save_data:
        input_ids_run_path = os.path.join(opt.model_dir, "data", "train_input_ids.txt")
        create_folder_if_not_exists(os.path.dirname(input_ids_run_path))

    pbar = tqdm(total=total_steps, desc="Training Progress", leave=False)

    while current_step < total_steps:
        total_loss = 0
        total_batches = 0

        for batch_idx, (input_ids, targets) in enumerate(train_loader):
            if current_step >= total_steps:
                break

            if opt.training2.save_data:
                with open(input_ids_run_path, "a") as f:
                    f.write(f"Step {current_step}, Batch {batch_idx}:\n")
                    f.write(" ".join(map(str, input_ids.tolist())) + "\n\n")

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

            current_step += 1
            batch_perplexity = math.exp(loss.item())
            pbar.set_description(
                f"Step {current_step}/{total_steps} - Batch PPL: {batch_perplexity:.2f}"
            )
            pbar.update(1)

            if current_step % opt.training2.save_every_n_steps == 0:
                save_checkpoint(model, optimizer, current_step, opt)

            if current_step % opt.training2.evaluate_every_n_steps == 0:
                avg_loss = total_loss / total_batches
                train_perplexity = math.exp(avg_loss)
                logging.info(
                    f"Step: {current_step} - Train Perplexity: {train_perplexity}"
                )
                train_perplexities.append(train_perplexity)
                valid_perplexity = test_model(model, opt, dataset="valid")
                logging.info(
                    f"Step: {current_step} - Valid Perplexity: {valid_perplexity}"
                )
                valid_perplexities.append(valid_perplexity)
                save_embeddings(model, current_step, opt)
                total_loss = 0
                total_batches = 0
                model.train()  # Set model back to training mode after evaluation

    pbar.close()
    test_perplexity = test_model(model, opt, dataset="test")
    logging.info(f"Test Perplexity: {test_perplexity}")
    plot_perplexity(train_perplexities, valid_perplexities, opt)
    save_loss(
        train_perplexities, valid_perplexities, test_perplexity, current_step, opt
    )
    return model


@torch.no_grad()
def test_model(model, opt, dataset="valid"):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    if dataset == "test":
        test_loader = opt.test_loader
    elif dataset == "valid":
        test_loader = opt.valid_loader
    elif dataset == "train":
        test_loader = opt.train_loader
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    total_loss = 0
    total_batches = 0
    total_samples = 0

    eval_steps = opt.training2.eval_steps
    if opt.training2.save_data:
        input_ids_run_path = os.path.join(opt.model_dir, "data", "val_input_ids.txt")
        create_folder_if_not_exists(os.path.dirname(input_ids_run_path))

    with torch.no_grad():
        pbar = tqdm(
            test_loader, desc=f"Evaluating on {dataset}", leave=False, total=eval_steps
        )
        for batch_idx, (input_ids, targets) in enumerate(pbar):
            if eval_steps is not None and total_batches >= eval_steps:
                break
            if opt.training2.save_data:
                with open(input_ids_run_path, "a") as f:
                    f.write(f"Step {0}, Batch {batch_idx}:\n")
                    f.write(" ".join(map(str, input_ids.tolist())) + "\n\n")

            input_ids = input_ids.to(opt.device)
            targets = targets.to(opt.device)
            input_mask = create_masks(input_ids)
            outputs = model(input_ids, input_mask)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            loss = criterion(outputs, targets)

            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_batches += 1
            total_samples += batch_size

            batch_perplexity = math.exp(loss.item())
            pbar.set_description(
                f"{dataset.capitalize()} Batch PPL: {batch_perplexity:.2f}"
            )

            if eval_steps is not None:
                pbar.update(1)

    avg_loss = total_loss / total_samples
    perplexity = math.exp(avg_loss)

    logging.info(
        f"{dataset.capitalize()} Perplexity: {perplexity:.2f} (evaluated on {total_samples} samples)"
    )

    return perplexity


def plot_perplexity(train_perplexities, valid_perplexities, opt):
    plt.figure()
    xticks = range(
        opt.training2.evaluate_every_n_steps,
        opt.training2.total_steps + 1,
        opt.training2.evaluate_every_n_steps,
    )
    plt.plot(xticks, train_perplexities, label="Train Perplexity")
    plt.plot(xticks, valid_perplexities, label="Valid Perplexity")
    if opt.core.plot_title:
        plt.title(opt.core.plot_title)
    else:
        plt.title(
            f"Experiment {opt.core.experiment_id} Model {opt.model_id} Perplexity"
        )
    plt.xlabel("Steps")
    plt.ylabel("Perplexity")
    plt.legend()
    path = os.path.join(opt.out_dir, f"Model {opt.model_id} Perplexity.png")
    logging.info(f"Saving to {path}")
    plt.savefig(path)


def dataloader_testing(opt):
    input_ids_run_path = os.path.join(opt.model_dir, "data", "input_ids.txt")
    create_folder_if_not_exists(os.path.dirname(input_ids_run_path))
    seed_all(opt.core.training_seed)
    with open(input_ids_run_path, "w") as f:
        for epoch in range(1, 3):
            for batch_idx, (input_ids, targets) in enumerate(tqdm(opt.train_loader)):
                f.write(f"Epoch {epoch}, Batch {batch_idx}:\n")
                f.write(" ".join(map(str, input_ids.tolist())) + "\n\n")

    return input_ids_run_path


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def experiment(opt):
    opt.exp_dir = os.path.join(opt.save_dir, "experiments", str(opt.core.experiment_id))
    # opt.exp_dir = f"experiments/{opt.core.experiment_id}"
    os.makedirs(opt.exp_dir, exist_ok=True)
    opt.log_path = os.path.join(opt.exp_dir, f"model{opt.model_id}.log")
    setup_logging(opt)
    log_cfg(opt)

    opt.model_dir = os.path.join(opt.exp_dir, "models", str(opt.model_id))
    opt.wte_dir = os.path.join(opt.model_dir, "wte")
    opt.ckpt_dir = os.path.join(opt.model_dir, "ckpts")
    opt.out_dir = os.path.join(opt.model_dir, "out")

    if os.path.exists(opt.model_dir):
        logging.warning(
            f"Experiment {opt.core.experiment_id} model {opt.model_id} already exists"
        )
        logging.warning("Do you wish to overwrite?")
        logging.warning("y/n")
        response = input("Do you wish to overwrite? (y/n): ")
        if response.lower() == "y":
            logging.warning("Overwriting")
            rm_dir(opt.model_dir)
        else:
            logging.warning("Exiting")
            sys.exit()

    os.makedirs(opt.model_dir)
    os.makedirs(opt.wte_dir)
    os.makedirs(opt.ckpt_dir)
    os.makedirs(opt.out_dir)
    logging.info(f"Experiment directory created ({opt.model_dir})")

    exp_str = (
        "=" * 10
        + f" Running Experiment {opt.core.experiment_id} Model {opt.model_id} "
        + "=" * 10
    )
    border = "=" * len(exp_str)
    logging.info(border)
    logging.info(exp_str)
    logging.info(border)

    start_time = time.time()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    opt.vocab_size = tokenizer.vocab_size  # 50,257 from GPT2
    train_text_path = os.path.join(opt.save_dir, "data", "wiki2.train.txt")
    valid_text_path = os.path.join(opt.save_dir, "data", "wiki2.valid.txt")
    test_text_path = os.path.join(opt.save_dir, "data", "wiki2.test.txt")

    train_text = read_corpus(
        train_text_path, tokenizer, first_n=opt.training2.dev_subset
    )
    valid_text = read_corpus(
        valid_text_path, tokenizer, first_n=opt.training2.dev_subset
    )
    test_text = read_corpus(test_text_path, tokenizer, first_n=opt.training2.dev_subset)
    wiki_train = WikiDataset(opt.model.seqlen, train_text, overlapping=True)
    g = torch.Generator()
    g.manual_seed(opt.core.training_seed)
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

    model = init_model(opt)
    opt.optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.training1.lr, betas=(0.9, 0.98), eps=1e-9
    )

    if opt.core.lock_weights:
        assert opt.core.starter_model_path
        load_model(model, opt.core.starter_model_path)

        init_wte(
            model,
            opt.core[f"model{opt.model_id}_embed_init"],
            opt.core[f"model{opt.model_id}_embed_init_seed"],
        )
        freeze_non_wte_weights(model)

    # count parameters
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Total trainable parameters: {total_trainable_params}")

    if opt.run.test_dataloader:
        dataloader_testing(opt)
    else:  # training
        model = train_model(model, opt.optimizer, opt, model_id=opt.model_id)

    logging.info(f"Time taken: {time.time() - start_time}")


def main():

    if len(sys.argv) < 2:
        raise ValueError("specify config name")
    experiment_num = sys.argv[1]
    model_id = int(sys.argv[2])

    opt = load_experiment_config(experiment_num)
    seed_all(opt.core.training_seed)

    opt.model_id = model_id
    experiment(opt)


if __name__ == "__main__":

    main()
