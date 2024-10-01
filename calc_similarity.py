import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def load_wtes(opt, step):
    path_fmt = "experiments/{}/models/{}/wte/wte_step{}.pt"
    wte1_path = path_fmt.format(opt.experiment_id, 1, step)
    wte2_path = path_fmt.format(opt.experiment_id, 2, step)
    wte1 = torch.load(wte1_path, weights_only=True)
    wte2 = torch.load(wte2_path, weights_only=True)
    return wte1, wte2


def compute_knn(embeddings: np.ndarray, k: int, metric: str = "inner_product"):
    """
    Compute the k-nearest neighbors for each token in the embedding matrix.

    Parameters:
    -----------
    embeddings : np.ndarray
        A NumPy array of shape (n_tokens, embedding_dim) containing the token embeddings.
    k : int
        The number of nearest neighbors to find for each token.
    metric : str, optional
        The distance metric to use. Supported metrics:
        - 'euclidean': Euclidean distance
        - 'cosine': Cosine distance
        - 'inner_product': Inner product similarity

    Returns:
    --------
    neighbors_indices : np.ndarray
        An array of shape (n_tokens, k) containing the indices of the k-nearest neighbors for each token.
    neighbors_distances_or_similarities : np.ndarray
        An array of shape (n_tokens, k) containing the corresponding distances or similarities.

    Raises:
    -------
    ValueError:
        If an unsupported metric is provided.
    """
    if metric not in ["euclidean", "cosine", "inner_product"]:
        raise ValueError(
            f"Unsupported metric '{metric}'. Supported metrics are 'euclidean', 'cosine', and 'inner_product'."
        )

    n_tokens = embeddings.shape[0]

    if metric in ["euclidean", "cosine"]:
        # Initialize NearestNeighbors with the specified metric
        # Note: 'cosine' metric in NearestNeighbors uses 1 - cosine similarity
        nbrs = NearestNeighbors(
            n_neighbors=k + 1,  # +1 to exclude the point itself
            metric=metric,
            algorithm="auto",
            n_jobs=-1,
        )
        nbrs.fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)

        # Exclude the first neighbor (itself)
        neighbors_indices = indices[:, 1 : k + 1]
        neighbors_distances_or_similarities = distances[:, 1 : k + 1]

        if metric == "cosine":
            # Convert cosine distance to cosine similarity
            neighbors_distances_or_similarities = (
                1 - neighbors_distances_or_similarities
            )

    elif metric == "inner_product":
        # Compute the inner product matrix
        inner_product_matrix = np.dot(embeddings, embeddings.T)

        # To exclude self, set the diagonal to -inf
        np.fill_diagonal(inner_product_matrix, -np.inf)

        # For each row, find the indices of the top k values
        neighbors_indices = np.argpartition(-inner_product_matrix, kth=k, axis=1)[:, :k]

        # Retrieve the corresponding inner product values
        # However, argpartition does not guarantee sorted order
        # So, we sort the top k in descending order
        row_indices = np.arange(n_tokens)[:, None]
        top_k_values = inner_product_matrix[row_indices, neighbors_indices]
        sorted_k_idx = np.argsort(-top_k_values, axis=1)
        neighbors_indices = neighbors_indices[row_indices, sorted_k_idx]
        neighbors_distances_or_similarities = top_k_values[row_indices, sorted_k_idx]

    return neighbors_indices, neighbors_distances_or_similarities


def compute_knn_similarity(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    k: int,
    metric1: str = "euclidean",
    metric2: str = "euclidean",
):
    """
    Compute the similarity between the k-nearest neighbors of each token in two embedding spaces.

    The similarity is defined as the size of the intersection of the KNN sets divided by k.

    Parameters:
    -----------
    embeddings1 : np.ndarray
        First embedding matrix of shape (n_tokens, embedding_dim1).
    embeddings2 : np.ndarray
        Second embedding matrix of shape (n_tokens, embedding_dim2).
    k : int
        The number of nearest neighbors to consider.
    metric1 : str, optional
        The distance metric to use for the first embedding space.
        Supported metrics: 'euclidean', 'cosine', 'inner_product'.
        Default is 'euclidean'.
    metric2 : str, optional
        The distance metric to use for the second embedding space.
        Supported metrics: 'euclidean', 'cosine', 'inner_product'.
        Default is 'euclidean'.

    Returns:
    --------
    similarity_scores : np.ndarray
        An array of shape (n_tokens,) containing the similarity scores for each token.

    Raises:
    -------
    ValueError:
        If the number of tokens in both embedding matrices does not match.
    """
    if embeddings1.shape[0] != embeddings2.shape[0]:
        raise ValueError(
            "Both embedding matrices must have the same number of tokens (rows)."
        )

    n_tokens = embeddings1.shape[0]

    # Compute KNN for both embeddings
    print("Computing KNN for the first embedding space...")
    knn_indices1, _ = compute_knn(embeddings1, k, metric=metric1)
    print("Computing KNN for the second embedding space...")
    knn_indices2, _ = compute_knn(embeddings2, k, metric=metric2)

    # Initialize similarity scores array
    similarity_scores = np.empty(n_tokens, dtype=np.float32)

    # Compute similarity for each token with progress bar
    print("Computing similarity between KNN sets...")
    for i in tqdm(range(n_tokens), desc="Processing tokens", unit="token"):
        set1 = set(knn_indices1[i])
        set2 = set(knn_indices2[i])
        intersection_size = len(set1.intersection(set2))
        similarity_scores[i] = intersection_size / k

    return similarity_scores


# def plot_knn_hist(knn_values, opt, step):
#
#     fig, ax = plt.subplots()
#     plt.ylim(0, opt.hist_y_max)
#     sns.histplot(knn_values, bins=10, kde=False, ax=ax)
#     plt.title(
#         f"Histogram of KNN Match Percentage with {opt.distance_metric} k={opt.k} step={step}"
#     )
#     save_path = f"experiments/{opt.experiment_id}/viz/knn_sim_hist/knn_sim_hist_{opt.distance_metric}_step_{step}.png"
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path)
#     plt.close()
#
#
# def plot_knn_line(opt, mean_knn_values):
#     fig, ax = plt.subplots()
#     plt.ylim(0, opt.line_y_max)
#     plt.plot(opt.steps, mean_knn_values)
#     plt.title(f"KNN Match Percentage with {opt.distance_metric} k={opt.k} step={step}")
#     save_path = f"experiments/{opt.experiment_id}/viz/knn_sim_line/knn_sim_line_{opt.distance_metric}.png"
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path)
#     plt.close()


def plot_knn_hist_and_line(knn_values, mean_knn_values, opt, step):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Plot histogram
    sns.histplot(
        knn_values,
        bins=10,
        kde=False,
        ax=ax1,
        color="skyblue",
        edgecolor="black",
        alpha=0.7,
    )

    # Set bottom x-label (KNN Match Percentage)
    ax1.set_xlabel("KNN Match Percentage", fontsize=14, color="blue", labelpad=10)
    ax1.xaxis.set_label_position("bottom")

    ax1.set_ylabel("Frequency", fontsize=14, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, opt.hist_y_max)

    # Create a twin x-axis for the top "Step" label
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xlabel("Step", fontsize=14, color="red", labelpad=10)

    # Create a twin y-axis for the line graph
    ax2 = ax1.twinx()
    ax2.set_ylim(0, opt.line_y_max)
    ax2.set_ylabel("Mean KNN Match Percentage", fontsize=14, color="red")
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.yaxis.tick_right()

    # Normalize steps to match the x-axis range of the histogram
    steps_up_to_current = opt.steps[: opt.steps.index(step) + 1]
    steps_normalized = [s / opt.steps[-1] for s in steps_up_to_current]
    mean_knn_values_up_to_current = mean_knn_values

    # Plot line graph
    ax2.plot(
        steps_normalized,
        mean_knn_values_up_to_current,
        color="red",
        marker="o",
        linestyle="-",
        linewidth=2,
        label="Mean KNN Match",
    )

    # Draw a dot on the most recent part of the line
    ax2.plot(
        steps_normalized[-1], mean_knn_values_up_to_current[-1], "ro", markersize=10
    )

    # Set xticks to correspond to the histogram plot
    ax1.set_xticks(np.linspace(0, 1, 11))  # xticks from 0 to 1
    ax2.set_xticks(ax1.get_xticks())
    ax1_top.set_xticks(ax1.get_xticks())

    # Set xticklabels
    ax1.set_xticklabels([f"{t:.1f}" for t in ax1.get_xticks()], fontsize=12)
    ax1_top.set_xticklabels(
        [int(t * opt.steps[-1]) for t in ax1.get_xticks()], fontsize=12
    )

    ax1_top.tick_params(axis="x", labelrotation=0, labelsize=12, colors="red")
    ax1.tick_params(axis="x", labelrotation=45, labelsize=12, colors="blue")

    # Add legends
    ax1.legend(["KNN Match Percentage Histogram"], loc="upper left", fontsize=12)
    ax2.legend(["Mean KNN Match Percentage"], loc="upper right", fontsize=12)

    # Set title
    plt.title(
        f"Experiment {opt.experiment_id} KNN Match Percentage\n"
        f"Distance Metric: {opt.distance_metric}, k={opt.k}, Step={step}",
        fontsize=16,
    )

    # Adjust layout to prevent clipping
    plt.tight_layout()
    save_path = f"experiments/{opt.experiment_id}/out/knn_sim_hist_line/knn_sim_hist_line_{opt.distance_metric}_step_{step}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def create_gif(image_paths, output_file, frame_duration, last_frame_duration):
    frames = [Image.open(image_path) for image_path in image_paths]

    # Prepare durations list
    durations = [frame_duration] * (len(frames) - 1) + [last_frame_duration]

    frames[0].save(
        output_file,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=durations,
        loop=0,
    )
    print(f"GIF created successfully: {output_file}")


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
            "experiment_id": 1,
            "k": 10,
            "distance_metric": "inner_product",
            # "steps": range(0, 16000 + 1, 4000),
            "steps": range(0, 16000 + 1, 1000),
            "hist_y_max": 50257 + 2000,
            "line_y_max": 0.40,
            "compute_knn_sim": True,
        }
    )
    # opt.make_vars(
    #     {
    #         "experiment_id": 2,
    #         "k": 10,
    #         "distance_metric": "inner_product",
    #         # "steps": range(0, 16000 + 1, 4000),
    #         "steps": range(0, 8000 + 1, 1000),
    #         "hist_y_max": 50257 + 2000,
    #         "line_y_max": 0.40,
    #         "compute_knn_sim": True,
    #     }
    # )

    mean_knn_values = []
    for step in opt.steps:
        if opt.compute_knn_sim:
            print(
                f"Loading weight matrices for experiment {opt.experiment_id} at step {step}..."
            )
            wte1, wte2 = load_wtes(opt, step)
            wte1, wte2 = wte1.numpy(), wte2.numpy()

            print("Starting KNN similarity computation...")
            similarities = compute_knn_similarity(
                wte1,
                wte2,
                opt.k,
                metric1=opt.distance_metric,
                metric2=opt.distance_metric,
            )
            # print("KNN Similarity (Euclidean):\n", similarity_euclidean)
            print("Mean KNN Similarity:\n", np.mean(similarities))
            save_path = f"experiments/{opt.experiment_id}/out/knn_sims/knn_sim_{opt.distance_metric}_step_{step}.npy"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, similarities)

        knn_values = np.load(
            f"experiments/{opt.experiment_id}/out/knn_sims/knn_sim_{opt.distance_metric}_step_{step}.npy"
        )
        mean_knn_val = np.mean(knn_values)
        mean_knn_values.append(mean_knn_val)
        print(f"Step {step} Mean KNN Sim {mean_knn_val}")
        plot_knn_hist_and_line(knn_values, mean_knn_values, opt, step)

    # save the mean_knn_values
    if not os.path.exists(f"experiments/{opt.experiment_id}/out/knn_sims"):
        os.makedirs(f"experiments/{opt.experiment_id}/out/knn_sims")
    np.save(
        f"experiments/{opt.experiment_id}/out/knn_sims/mean_knn_values_{opt.distance_metric}.npy",
        mean_knn_values,
    )
    opt.gif_path = f"experiments/{opt.experiment_id}/out/knn_sim_hist_line/knn_sim_line_{opt.distance_metric}.gif"
    if not os.path.exists(os.path.dirname(opt.gif_path)):
        os.makedirs(os.path.dirname(opt.gif_path))
    image_paths = []
    for step in opt.steps:
        sim_viz_path = f"experiments/{opt.experiment_id}/out/knn_sim_hist_line/knn_sim_hist_line_{opt.distance_metric}_step_{step}.png"
        image_paths.append(sim_viz_path)
    create_gif(image_paths, opt.gif_path, 850, 3000)
