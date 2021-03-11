import sys
from argparse import ArgumentParser

sys.path.append('../../analysis')
import os
from glob import glob
from pathlib import Path

import torch
from pl_bolts.models.regression import LogisticRegression
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
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
        return x, self.labels[index], self.files[index]

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
    log_file = 'results.csv'
    if not os.path.exists(log_file):
        with open(log_file, 'w') as log:
            log.write('val_acc;test_acc;config\n')

    print('loading data...')
    data_module = GramsDataModule(data_root=args.data_root,
                                  cats_dirs=args.cats_dirs,
                                  num_workers=8)
    model = LogisticRegression(input_dim=data_module.dims,
                               num_classes=data_module.num_classes,
                               learning_rate=1e-4,
                               l2_strength=args.l2)

    checkpoint_callback = ModelCheckpoint(
        filepath=f'checkpoints/version_{args.trial}/best',
        verbose=True,
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )

    trainer = Trainer(checkpoint_callback=checkpoint_callback,
                      gpus=[0],
                      max_epochs=10)

    trainer.fit(model=model,
                datamodule=data_module)

    val_acc = trainer.callback_metrics['val_acc']

    result = trainer.test()
    test_acc = result[0]['test_acc']

    with open(log_file, 'a') as log:
        log.write(f'{val_acc};{test_acc};{vars(args)}\n')
        log.flush()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--cats_dirs', nargs='+')
    parser.add_argument('--l2', type=float)
    parser.add_argument('--version', type=int)
    parser.add_argument('--trial', type=int)

    args = parser.parse_args()
    main(args)
