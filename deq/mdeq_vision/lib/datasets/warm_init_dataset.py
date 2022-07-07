import collections
from pathlib import Path

import torch
from torch.utils.data import Dataset


class WarmInitDataset(Dataset):
    def __init__(self, dataset, warm_init_path):
        super().__init__()
        self.internal_dataset = dataset
        if warm_init_path:
            self.warm_init_path = Path(warm_init_path)
        else:
            self.warm_init_path = None

    def __getitem__(self, index):
        data = self.internal_dataset[index]
        warm_init = None
        if self.warm_init_path:
            warm_file = self.warm_init_path / f'{index}.pt'
            if warm_file.exists():
                warm_init = torch.load(warm_file)
        return (*data, warm_init, index)

    def __len__(self):
        return len(self.internal_dataset)


def collate_fn_none(batch):
    try:
        return torch.utils.data.default_collate(batch)
    except TypeError:
        elem = batch[0]
        if isinstance(elem, collections.abc.Sequence):
            transposed = list(zip(*batch))
            return [collate_fn_none(samples) for samples in transposed]
        if any(e is None for e in batch):
            return None
