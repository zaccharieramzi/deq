import pathlib


from pathlib import Path

import torch
from torch.utils.data import Dataset


class WarmInitDataset(Dataset):
    def __init__(self, dataset, warm_init_path):
        super().__init__()
        self.internal_dataset = dataset
        self.warm_init_path = Path(warm_init_path)

    def __getitem__(self, index):
        data = self.internal_dataset[index]
        warm_file = self.warm_init_path / f'{index}.pt'
        if warm_file.exists():
            warm_init = torch.load(warm_file)
        else:
            warm_init = None
        return (*data, warm_init, index)

    def __len__(self):
        return len(self.internal_dataset)
