import torch
from torch.utils.data import Dataset


def read_corpus(filename, tokenizer, first_n=None):
    seq = []
    with open(filename, "rt") as f:
        for i, line in enumerate(f):
            if first_n and i > first_n:
                return seq
            line = line.replace("\n", "")
            tokens = tokenizer(line)
            for t in tokens["input_ids"]:
                seq.append(t)
    return seq


class WikiDataset(Dataset):
    def __init__(self, seqlen, data, overlapping=True):
        super().__init__()
        self.seqlen = seqlen
        self.data = data
        self.overlapping = overlapping

    def __getitem__(self, idx):
        if self.overlapping:
            return torch.tensor(self.data[idx : idx + self.seqlen]), torch.tensor(
                self.data[idx + 1 : idx + 1 + self.seqlen]
            )
        else:
            return torch.tensor(
                self.data[idx * self.seqlen : (idx + 1) * self.seqlen]
            ), torch.tensor(
                self.data[idx * self.seqlen + 1 : (idx + 1) * self.seqlen + 1]
            )

    def __len__(self):
        if self.overlapping:
            return len(self.data) - self.seqlen
        else:
            return len(self.data) // self.seqlen


def create_masks(input_ids):
    seq_len = input_ids.size(1)
    batch_size = input_ids.size(0)
    mask = ~torch.triu(
        torch.ones((seq_len, seq_len), device=input_ids.device), diagonal=1
    ).bool()
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask
