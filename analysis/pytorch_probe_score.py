import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pl_bolts.models.regression import LogisticRegression
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.dataset import random_split

from util import Grams


class GramLayerDataset(Dataset):
    def __init__(self, grams, labels):
        super(GramLayerDataset, self).__init__()
        self.grams = torch.tensor(grams, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, index: int):
        return self.grams[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.grams)

def probe_score(grams, batch_size=4096, fast_dev_run=False):
    layers = []
    accuracies = []
    stds = []

    for layer, X in zip(grams.layer_names, grams):
        X = StandardScaler().fit_transform(X)
        folds = StratifiedKFold(n_splits=5, shuffle=True).split(X, grams.labels)
        accs = []
        for split in folds:
            train_idx, test_idx = split

            dset = GramLayerDataset(X[train_idx], grams.labels[train_idx])
            test_dset = GramLayerDataset(X[test_idx], grams.labels[test_idx])

            train_dset, val_dset = random_split(dset, lengths=[int(len(dset) * 0.9), len(dset) - int(len(dset) * 0.9)])

            train_loader = DataLoader(dataset=train_dset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=8)

            val_loader = DataLoader(dataset=val_dset,
                                    batch_size=batch_size,
                                    drop_last=False,
                                    num_workers=8)

            test_loader = DataLoader(dataset=test_dset,
                                     batch_size=batch_size,
                                     drop_last=False,
                                     num_workers=8)

            model = LogisticRegression(input_dim=X.shape[-1],
                                       learning_rate=1e-3,
                                       l2_strength=1e-4,
                                       num_classes=len(set(grams.labels)))

            # fit
            early_stopping = EarlyStopping(monitor='val_ce_loss',
                                           patience=20)

            trainer = pl.Trainer(gpus=[0],
                                 check_val_every_n_epoch=5,
                                 callbacks=[early_stopping],
                                 fast_dev_run=fast_dev_run)
            trainer.fit(model=model,
                        train_dataloader=train_loader,
                        val_dataloaders=val_loader)

            result = trainer.test(test_dataloaders=test_loader)
            accs.append(result[0]['test_acc'])

        layers.append(layer)
        accuracies.append(np.mean(accs))
        stds.append(np.std(accs))

    df = pd.DataFrame({
        'layer': layers,
        'linear_probe': accuracies,
        'linear_probe_err': stds
    })
    print(df)
    return df


if __name__ == '__main__':
    grams = Grams('uvnet_data/solidmnist_all_fnorm')
    probe_score(grams)

