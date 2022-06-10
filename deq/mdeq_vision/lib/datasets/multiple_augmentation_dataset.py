import torch
from torch.utils.data import Dataset


class MultiAugmentationDataset(Dataset):
    def __init__(self, dataset, augment, n_augment=2):
        super().__init__()
        self.dataset = dataset
        self.augment = augment
        self.n_augment = n_augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        augments = [
            self.augment(image).unsqueeze(0)
            for _ in range(self.n_augment)
        ]
        sub_batch = torch.cat(augments, dim=0)
        label = torch.tensor(label).unsqueeze(0).repeat(self.n_augment)
        return sub_batch, label
