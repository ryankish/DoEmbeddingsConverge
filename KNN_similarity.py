from embedding_translator import LinearRegressionMapper
import os
import re
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, linear_kernel
from tqdm import tqdm
import seaborn as sns

def get_embed_weights_files(experiment_id, model_id, epoch):
    target_directory = os.path.join('experiments', str(experiment_id), 'models', str(model_id), 'embeddings')
    try:
        files = os.listdir(target_directory)
    except FileNotFoundError:
        return []
    
    # Filter out files that match the pattern 'embed_weights_epoch_<number>.pt'
    pattern = re.compile(r'embed_weights_epoch_(\d+)\.pt')
    epoch_files = [f for f in files if pattern.match(f) and int(pattern.search(f).group(1)) == epoch]
    
    return [os.path.join(target_directory, f) for f in epoch_files]

def load_weights_as_tensors(weights_path):
    return torch.load(weights_path)

def extract_epoch_number(filename):
    pattern = re.compile(r'embed_weights_epoch_(\d+)\.pt')
    match = pattern.search(filename)
    if match:
        return int(match.group(1))
    else:
        return None

def load_embeddings_for_epoch(opt, epoch):
    embedding1_files = get_embed_weights_files(opt.experiment_id, 1, epoch)
    embedding2_files = get_embed_weights_files(opt.experiment_id, 2, epoch)
    
    assert len(embedding1_files) == len(embedding2_files), "Different number of epoch files for model 1 and model 2"
    
    embeddings1 = [load_weights_as_tensors(f) for f in embedding1_files]
    embeddings2 = [load_weights_as_tensors(f) for f in embedding2_files]
    
    return embeddings1[0], embeddings2[0]

def knn_match_percentage_one_by_one(embedding1, embedding2, k, distance_metric):
    if distance_metric == 'euclidean':
        distances = euclidean_distances(embedding1, embedding2)
    elif distance_metric == 'cosine':
        distances = 1 - cosine_similarity(embedding1, embedding2)
    elif distance_metric == 'inner_product':
        distances = -linear_kernel(embedding1, embedding2)
    else:
        raise ValueError("Invalid distance metric. Choose from 'euclidean', 'cosine', 'inner_product'")

    matches_per_token = []

    for i in tqdm(range(embedding1.shape[0]), desc="Calculating KNN similarity"):
        knn_indices_1 = np.argsort(distances[i])[:k]
        knn_indices_2 = np.argsort(distances[:, i])[:k]
        matches = len(set(knn_indices_1).intersection(set(knn_indices_2)))
        matches_per_token.append(matches)
    
    match_percentage_per_token = [(matches / k) for matches in matches_per_token]
    
    return match_percentage_per_token

class Options:
    def __init__(self) -> None:
        pass

    def make_vars(self, args: dict):
        for key, val in args.items():
            self.__setattr__(key, val)

if __name__ == '__main__':
    opt = Options()
    opt.make_vars({
        'experiment_id': 1,
        'distance_metric': 'inner_product',
        'k': 10,
        'epochs': [0, 40],
    })

    for epoch in opt.epochs:
        embedding1, embedding2 = load_embeddings_for_epoch(opt, epoch)
        #print(embedding1.shape[0])
        # knn_values = knn_match_percentage_one_by_one(embedding1.numpy(), embedding2.numpy(), opt.k, opt.distance_metric)
        
        # np.save(f'experiments/{opt.experiment_id}/knn_sim_{opt.distance_metric}_epoch_{epoch}.npy', knn_values)
        
        knn_values = np.load(f'experiments/{opt.experiment_id}/knn_sim_{opt.distance_metric}_epoch_{epoch}.npy')
        print('Mean KNN Sim', np.mean(knn_values))
        
        fig, ax = plt.subplots()
        plt.ylim(0, embedding1.shape[0])
        sns.histplot(knn_values, bins=10, kde=False, ax=ax)
        plt.title(f'Histogram of KNN Match Percentage with {opt.distance_metric} k={opt.k} epoch={epoch}')
        plt.savefig(f'experiments/{opt.experiment_id}/knn_sim_hist_{opt.distance_metric}_epoch_{epoch}.png')
