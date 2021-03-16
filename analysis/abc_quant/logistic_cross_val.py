import sys
from argparse import ArgumentParser

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

sys.path.append('../../analysis')
import os
from glob import glob
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset

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


def main(args):
    if not os.path.exists(args.log):
        with open(args.log, 'w') as log:
            log.write('mean_f1;std_f1;config;best_params\n')

    print(f'{args.version}: loading data...')
    print(args)

    dset = GramDataset(data_root=args.data_root, cats_dirs=args.cats_dirs)
    print(len(dset))
    if len(dset) > 4000:
        return

    np.random.seed(1234)
    X, y, _ = dset[shuffle(np.arange(len(dset)))]
    X = X.numpy()

    X = StandardScaler().fit_transform(X)
    del dset

    param_grid = {'C': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]}

    grid = GridSearchCV(LogisticRegression(max_iter=2000, class_weight='balanced'), scoring='f1_weighted', param_grid=param_grid, cv=5, n_jobs=8, verbose=1)
    grid.fit(X, y)

    preds = grid.best_estimator_.predict(X)
    cf = confusion_matrix(y, preds)

    print(args.cats_dirs)
    print(grid.cv_results_['mean_test_score'][grid.best_index_])
    print(cf)

    test_acc = grid.cv_results_['mean_test_score'][grid.best_index_]
    test_std = grid.cv_results_['std_test_score'][grid.best_index_]

    with open(args.log, 'a') as log:
        log.write(f'{test_acc};{test_std};{vars(args)};{grid.best_params_}\n')
        log.flush()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--log', type=str, default='results_logistic_cross_val_no_pca.csv')
    # parser.add_argument('--data_root', type=str, default='../uvnet_data/abc_sub_mu_only')
    parser.add_argument('--data_root', type=str, default='../psnet_data/abc_all')
    parser.add_argument('--cats_dirs', nargs='+',
                        default=['abc_quant_data/wheel_v_cog/wheel', 'abc_quant_data/wheel_v_cog/cog'])
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--trial', type=int, default=0)

    args = parser.parse_args()
    main(args)
