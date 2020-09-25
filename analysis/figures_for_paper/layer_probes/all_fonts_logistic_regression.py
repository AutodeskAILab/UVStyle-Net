import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import Adam

sys.path.append('../../../analysis')

from all_fonts_svms.all_fonts_svm import pca_reduce
from util import Grams, weight_layers


def eval(model, X, y):
    linear.eval()
    preds = torch.softmax(model(X), dim=-1)
    preds = preds.argmax(dim=-1)
    return accuracy_score(y.detach().cpu().numpy(), preds.detach().cpu().numpy())


if __name__ == '__main__':
    device = torch.device('cuda:0')
    grams = Grams('../uvnet_data/solidmnist_filtered_font_families')
    reduced = pca_reduce(grams, 70, '../cache/solidmnist_filtered_font_families')
    print(len(set(grams.labels)), ' classes - random baseline:', 1 / len(set(grams.labels)))

    n_splits = 5
    layer_weights = np.eye(len(grams.layer_names))  # type:np.ndarray

    scores = []
    for weights in layer_weights:
        weighted = weight_layers(reduced, weights)
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True).split(weighted, grams.labels)

        for train_val_idx, test_idx in folds:
            linear = torch.nn.Linear(weighted.shape[-1], len(set(grams.labels)), bias=True).to(device)
            crit = CrossEntropyLoss()
            optimizer = Adam(linear.parameters(), lr=5e-3, weight_decay=1e-5)

            x_train_val, x_test = torch.tensor(weighted[train_val_idx]), torch.tensor(weighted[test_idx])
            y_train_val, y_test = torch.tensor(grams.labels[train_val_idx], dtype=torch.long), torch.tensor(
                grams.labels[test_idx], dtype=torch.long)

            train_idx, val_idx = next(StratifiedShuffleSplit(n_splits=1, test_size=0.1).split(x_train_val, y_train_val))

            x_train = x_train_val[train_idx].to(device)
            y_train = y_train_val[train_idx].to(device)
            x_val = x_train_val[val_idx].to(device)
            y_val = y_train_val[val_idx].to(device)

            best_val = 0.
            count = 0
            for epoch in range(5000):
                linear.train()
                optimizer.zero_grad()
                # Forward pass
                y_pred = linear(x_train)
                # Compute Loss
                loss = crit(y_pred, y_train)
                # Backward pass
                loss.backward()
                optimizer.step()

                if epoch % 100 == 0:
                    val_score = eval(linear, x_val, y_val)
                    print('val:', val_score, 'loss:', loss.__float__())
                    if val_score >= best_val:
                        best_val = val_score
                        count = 0
                    else:
                        count += 1
                        if count > 1:
                            print('done in ', epoch, 'epochs')
                            break

            score = eval(linear.to(device), x_test.to(device), y_test.to(device))
            print('layer', weights.argmax(), 'test acc:', score)
            scores.append(score)

    pd.DataFrame({
        'layer': layer_weights.argmax(axis=-1).repeat(n_splits).tolist(),
        'score': scores
    }).to_csv('all_fonts_logistic_regression_results.csv', index=False, sep=',')
