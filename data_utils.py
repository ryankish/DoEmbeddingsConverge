import torch
from torch.utils.data import Dataset, DataLoader


def read_corpus(filename,tokenizer,first_n=None):
    seq = []
    with open(filename,'rt') as f:
        for i, line in enumerate(f):
            if first_n and i > first_n:
                return(seq)
            line = line.replace('\n','')
            tokens = tokenizer(line)
            for t in tokens['input_ids']:
                seq.append(t)
    return(seq)

class WikiDataset(Dataset):
    def __init__(self, opt, data, overlapping=True, seed=0):
        super().__init__()
        self.opt = opt
        self.data = data
        self.overlapping = overlapping
        self.seed = seed
        if self.seed is not None:
            self.shuffle_data()

    def shuffle_data(self):
        rng = torch.Generator()
        rng.manual_seed(self.seed)
        indices = torch.randperm(len(self.data), generator=rng)
        self.data = [self.data[i] for i in indices]

    def __getitem__(self, idx):
        if self.overlapping:
            return torch.tensor(self.data[idx:idx+self.opt.seqlen]), \
                torch.tensor(self.data[idx+1:idx+1+self.opt.seqlen])
        else:
            return torch.tensor(self.data[idx*self.opt.seqlen:(idx+1)*self.opt.seqlen]), \
                torch.tensor(self.data[idx*self.opt.seqlen+1:(idx+1)*self.opt.seqlen+1])

    def __len__(self):
        if self.overlapping:
            return len(self.data) - self.opt.seqlen
        else:
            return len(self.data) // self.opt.seqlen


def create_masks(input_ids):
    seq_len = input_ids.size(1)
    batch_size = input_ids.size(0)
    mask = ~torch.triu(torch.ones((seq_len, seq_len), device=input_ids.device), diagonal=1).bool()
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask
