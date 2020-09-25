import pandas as pd
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, Dataset
from torch.utils.data.dataloader import DataLoader

from all_fonts_svms.all_fonts_svm import pca_reduce
from util import Grams, weight_layers


class EmbeddingsDataset(Dataset):
    def __init__(self,
                 data_root='../contrastive_data/rotation_only_filtered'):
        self.X = torch.tensor(np.load(data_root + '/embeddings.npy'), dtype=torch.float)
        graph_files = np.loadtxt(data_root + '/graph_files.txt', dtype=np.str, delimiter=',')
        names = map(lambda f: f[2:-10], graph_files)
        df = pd.DataFrame(names, columns=['name'])
        self.y = torch.tensor(df['name'].astype('category').cat.codes.values, dtype=torch.long)
        self.graph_files = graph_files

    def __getitem__(self, item):
        return self.X[item], self.y[item], self.graph_files[item]

    def __len__(self):
        return len(self.X)


class EmbeddingsDataModule(LightningDataModule):
    def __init__(self, batch_size=32, data_root='../contrastive_data/rotation_only_filtered'):
        super().__init__()
        self.batch_size = batch_size
        dset = EmbeddingsDataset(data_root=data_root)
        test_size = int(0.1 * len(dset))
        val_size = int(0.1 * len(dset))
        train_size = len(dset) - test_size - val_size
        self.train, self.val, self.test = random_split(dset, [train_size, val_size, test_size])

    def train_dataloader(self, *args, **kwargs):
        return DataLoader(self.train, self.batch_size, shuffle=True, collate_fn=collate, num_workers=16)

    def val_dataloader(self, *args, **kwargs):
        return DataLoader(self.val, self.batch_size, shuffle=False, collate_fn=collate, num_workers=16)

    def test_dataloader(self, *args, **kwargs):
        return DataLoader(self.test, self.batch_size, shuffle=False, collate_fn=collate, num_workers=16)


class GramsDataset(Dataset):
    def __init__(self,
                 data_root='../uvnet_data/solidmnist_filtered_font_families',
                 cache_file='filtered_grams_165'):
        grams = Grams(data_root=data_root)
        reduced = pca_reduce(grams, 165, f'../cache/{cache_file}')
        self.X = torch.tensor(weight_layers(reduced, np.ones(len(grams))), dtype=torch.float)
        self.y = torch.tensor(grams.labels.copy(), dtype=torch.long)
        self.graph_files = grams.graph_files

    def __getitem__(self, item):
        return self.X[item], self.y[item], self.graph_files[item]

    def __len__(self):
        return len(self.X)


def collate(batch):
    grams, labels, files = map(list, zip(*batch))
    labels = torch.stack(labels)
    grams = torch.stack(grams)
    return grams, labels, files


class GramsDataModule(LightningDataModule):
    def __init__(self, batch_size=32, data_root='../uvnet_data/solidmnist_filtered_font_families',
                 cache_file='filtered_grams'):
        super().__init__()
        self.batch_size = batch_size
        dset = GramsDataset(data_root=data_root,
                            cache_file=cache_file)
        test_size = int(0.1 * len(dset))
        val_size = int(0.1 * len(dset))
        train_size = len(dset) - test_size - val_size
        self.train, self.val, self.test = random_split(dset, [train_size, val_size, test_size])

    def train_dataloader(self, *args, **kwargs):
        return DataLoader(self.train, self.batch_size, shuffle=True, collate_fn=collate, num_workers=16)

    def val_dataloader(self, *args, **kwargs):
        return DataLoader(self.val, self.batch_size, shuffle=False, collate_fn=collate, num_workers=16)

    def test_dataloader(self, *args, **kwargs):
        return DataLoader(self.test, self.batch_size, shuffle=False, collate_fn=collate, num_workers=16)


if __name__ == '__main__':
    g = GramsDataset()
