import sys
from argparse import ArgumentParser

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.utils import shuffle

sys.path.append('../../analysis')
import os
from glob import glob
from pathlib import Path

import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

from util import Grams


class GramDataset(Dataset):

    def __init__(self, data_root, cats_dirs) -> None:
        super().__init__()

        self.files = []
        self.labels = []
        self.categories = [Path(cats_dir).stem for cats_dir in cats_dirs]

        files = []
        for cats_dir in cats_dirs:
            files += glob(cats_dir + '/*.png')

        self.grams = Grams(data_root)
        self.name_to_id = {name: id for id, name in enumerate(self.grams.graph_files)}

        cat_to_label = {cat: label for label, cat in enumerate(self.categories)}

        for file in files:
            path = Path(file)
            self.files.append(path.stem + '.bin')
            self.labels.append(cat_to_label[path.parent.stem])

    def __getitem__(self, index):
        num_layers = len(self.grams.layer_names)
        x = torch.cat([torch.tensor(self.grams[i][index]) for i in range(num_layers)], dim=-1)
        return x, np.array(self.labels)[index], np.array(self.files)[index]

    def __len__(self):
        return len(self.files)

    @staticmethod
    def collate(data):
        x, labels, _ = zip(*data)
        return torch.stack(x), torch.tensor(labels)


class GramsDataModule(LightningDataModule):
    def __init__(self, data_root, cats_dirs, val_size=0.1, test_size=0.1, batch_size=32, num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        dset = GramDataset(data_root=data_root,
                           cats_dirs=cats_dirs)
        self.dims = sum([dset.grams[i].shape[-1] for i in range(len(dset.grams))])
        self.num_classes = len(dset.categories)

        num_test = int(test_size * len(dset))
        num_val = int(val_size * len(dset))
        num_train = len(dset) - num_test - num_val
        self.train, self.val, self.test = random_split(dset, [num_train, num_val, num_test])

    def train_dataloader(self, *args, **kwargs):
        return DataLoader(self.train, self.batch_size, shuffle=True, collate_fn=GramDataset.collate,
                          num_workers=self.num_workers)

    def val_dataloader(self, *args, **kwargs):
        return DataLoader(self.val, self.batch_size, shuffle=False, collate_fn=GramDataset.collate,
                          num_workers=self.num_workers)

    def test_dataloader(self, *args, **kwargs):
        return DataLoader(self.test, self.batch_size, shuffle=False, collate_fn=GramDataset.collate,
                          num_workers=self.num_workers)


def main(args):
    if not os.path.exists(args.log):
        with open(args.log, 'w') as log:
            log.write('test_acc;config;best_params\n')

    print('loading data...')

    dset = GramDataset(data_root=args.data_root, cats_dirs=args.cats_dirs)

    np.random.seed(9876)
    X, y, _ = dset[shuffle(np.arange(len(dset)))]
    X = X.numpy()

    param_grid = {'C': [0.1, 1, 10, 100],
                  'gamma': [1, 0.1, 0.01, 0.001],
                  'kernel': ['poly']}

    grid = GridSearchCV(SVC(), scoring='accuracy', param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    grid.fit(X, y)

    test_acc = grid.cv_results_['mean_test_score'][grid.best_index_]
    test_std = grid.cv_results_['std_test_score'][grid.best_index_]

    with open(args.log, 'a') as log:
        log.write(f'{test_acc};{test_std};{vars(args)};{grid.best_params_}\n')
        log.flush()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--log', type=str)
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--cats_dirs', nargs='+')
    parser.add_argument('--version', type=int)
    parser.add_argument('--trial', type=int)

    args = parser.parse_args()
    main(args)
