import torch
from transformers import GPT2TokenizerFast
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def load_weights_as_tensors(weights_path):
    return torch.load(weights_path)

def calculate_token_frequencies(file_path, tokenizer, chunk_size=1000):
    counter = Counter()
    with open(file_path, 'r') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            tokens = tokenizer.tokenize(chunk)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            counter.update(token_ids)
    return counter

def calculate_embedding_norms(embeddings):
    norms = torch.norm(embeddings, dim=1).detach().numpy()
    return norms

def analyze_embeddings(opt, text_path):

    embeddings1 = load_weights_as_tensors(opt.embeddings1_path)
    embeddings2 = load_weights_as_tensors(opt.embeddings2_path)

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    token_frequencies = calculate_token_frequencies(text_path, tokenizer)

    all_token_indices = list(token_frequencies.keys())

    norms1 = calculate_embedding_norms(embeddings1)
    norms2 = calculate_embedding_norms(embeddings2)

    frequencies = np.array([token_frequencies[idx] for idx in all_token_indices])
    norms1 = norms1[all_token_indices]
    norms2 = norms2[all_token_indices]

    correlation1 = np.corrcoef(frequencies, norms1)[0, 1]
    correlation2 = np.corrcoef(frequencies, norms2)[0, 1]

    print(f"Correlation between frequency and norm for Model 1: {correlation1}")
    print(f"Correlation between frequency and norm for Model 2: {correlation2}")

    log_frequencies = np.log(frequencies)
    log_norms1 = np.log(norms1)
    log_norms2 = np.log(norms2)

    log_correlation1 = np.corrcoef(log_frequencies, log_norms1)[0, 1]
    log_correlation2 = np.corrcoef(log_frequencies, log_norms2)[0, 1]

    print(f"Log-Log Correlation between frequency and norm for Model 1: {log_correlation1}")
    print(f"Log-Log Correlation between frequency and norm for Model 2: {log_correlation2}")

    min_norm = min(norms1.min(), norms2.min())
    max_norm = max(norms1.max(), norms2.max())


    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(frequencies, norms1, alpha=0.5)
    plt.xlabel('Token Frequency')
    plt.ylabel('Embedding Norm')
    plt.title(f'{opt.name} 1: Frequency vs. Norm')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(min_norm, max_norm)

    plt.subplot(1, 2, 2)
    plt.scatter(frequencies, norms2, alpha=0.5)
    plt.xlabel('Token Frequency')
    plt.ylabel('Embedding Norm')
    plt.title(f'{opt.name} 2: Frequency vs. Norm')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(min_norm, max_norm)

    plt.tight_layout()

    plt.tight_layout()
    #plt.show()
    plt.savefig(f'experiments/{opt.experiment_id}/NormAnalysis.png')

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
        'name': 'Model'
    })
    opt.embeddings1_path = f'experiments/{opt.experiment_id}/embedding1.pt'
    opt.embeddings2_path = f'experiments/{opt.experiment_id}/translated_embedding2.pt'
    training_text_path = 'data/wiki2.train.txt'

    analyze_embeddings(opt, training_text_path)

    opt = Options()
    opt.make_vars({
        'experiment_id': 2,
        'name': 'Embedding Space'
    })
    opt.embeddings1_path = f'experiments/{opt.experiment_id}/embedding1.pt'
    opt.embeddings2_path = f'experiments/{opt.experiment_id}/translated_embedding2.pt'
    training_text_path = 'data/wiki2.train.txt'

    analyze_embeddings(opt, training_text_path)
