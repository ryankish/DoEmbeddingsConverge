import os
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    linear_kernel,
)
from tqdm import tqdm


def load_wtes(opt, step):
    path_fmt = "experiments/{}/models/{}/wte/wte_step{}.pt"
    wte1_path = path_fmt.format(opt.experiment_id, 1, step)
    wte2_path = path_fmt.format(opt.experiment_id, 2, step)
    wte1 = torch.load(wte1_path, weights_only=True)
    wte2 = torch.load(wte2_path, weights_only=True)
    return wte1, wte2


def tokenwise_knn_sim(wte1, wte2, k, distance_metric):
    if distance_metric == "euclidean":
        distances = euclidean_distances(wte1, wte2)
    elif distance_metric == "cosine":
        distances = 1 - cosine_similarity(wte1, wte2)
    elif distance_metric == "inner_product":
        distances = -linear_kernel(wte1, wte2)
    else:
        raise ValueError(
            "Invalid distance metric. Choose from 'euclidean', 'cosine', 'inner_product'"
        )

    matches_per_token = []

    for i in tqdm(range(wte1.shape[0]), desc="Calculating KNN similarity"):
        knn_indices_1 = np.argsort(distances[i])[:k]
        knn_indices_2 = np.argsort(distances[:, i])[:k]
        matches = len(set(knn_indices_1).intersection(set(knn_indices_2)))
        matches_per_token.append(matches)

    match_percentage_per_token = [(matches / k) for matches in matches_per_token]

    return match_percentage_per_token


def plot_knn_hist(knn_values, opt, step):

    fig, ax = plt.subplots()
    plt.ylim(0, opt.hist_y_max)
    sns.histplot(knn_values, bins=10, kde=False, ax=ax)
    plt.title(
        f"Histogram of KNN Match Percentage with {opt.distance_metric} k={opt.k} step={step}"
    )
    save_path = f"experiments/{opt.experiment_id}/viz/knn_sim_hist/knn_sim_hist_{opt.distance_metric}_step_{step}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_knn_line(opt, mean_knn_values):
    fig, ax = plt.subplots()
    plt.ylim(0, opt.line_y_max)
    plt.plot(opt.steps, mean_knn_values)
    plt.title(f"KNN Match Percentage with {opt.distance_metric} k={opt.k} step={step}")
    save_path = f"experiments/{opt.experiment_id}/viz/knn_sim_line/knn_sim_line_{opt.distance_metric}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


class Options:
    def __init__(self) -> None:
        pass

    def make_vars(self, args: dict):
        for key, val in args.items():
            self.__setattr__(key, val)


if __name__ == "__main__":
    opt = Options()
    opt.make_vars(
        {
            "experiment_id": 2,
            "distance_metric": "inner_product",
            "k": 10,
            "steps": range(0, 10000 + 1, 1000),
            "hist_y_max": 50257 + 1000,
            "line_y_max": 0.35,
            "compute_knn_sim": True,
        }
    )
    mean_knn_values = []
    for step in opt.steps:

        if opt.compute_knn_sim:
            wte1, wte2 = load_wtes(opt, step)
            knn_values = tokenwise_knn_sim(
                wte1.numpy(), wte2.numpy(), opt.k, opt.distance_metric
            )
            save_path = f"experiments/{opt.experiment_id}/knn_sims/knn_sim_{opt.distance_metric}_step_{step}.npy"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, knn_values)

        knn_values = np.load(
            f"experiments/{opt.experiment_id}/knn_sims/knn_sim_{opt.distance_metric}_step_{step}.npy"
        )
        mean_knn_val = np.mean(knn_values)
        mean_knn_values.append(mean_knn_val)
        print(f"Step {step} Mean KNN Sim {mean_knn_val}")

        plot_knn_hist(knn_values, opt, step)

    plot_knn_line(opt, mean_knn_values)
