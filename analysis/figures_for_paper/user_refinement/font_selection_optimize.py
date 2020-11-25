import os
import sys

import torch
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append('../../../analysis')

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from all_fonts_svms.all_fonts_svm import pca_reduce
from constrained_optimization import optimize
from util import Grams


def hits_at_k_score(X, weights, positives, k=10):
    grams_0 = torch.zeros(len(X[0]), 70)
    grams_0[:, :21] = torch.tensor(X[0].copy())
    grams_padded = torch.stack([grams_0] + [torch.tensor(gram.copy()) for gram in X[1:]])
    X = grams_padded.permute(1, 0, 2).to(device)  # shape: N x 7 x 70
    # X shape: N x 7 x 70
    scores = []
    for query in positives:
        q = X[query]
        layerwise_distances = 1 - torch.cosine_similarity(q[None, :, :], X, dim=-1)
        weighted = layerwise_distances * weights[None, :]
        distances = weighted.sum(-1)  # type: torch.Tensor
        neighbours = set(torch.argsort(distances)[:k].detach().cpu().numpy())
        score = len(neighbours.intersection(set(positives))) / k
        scores.append(score)
    return np.mean(scores), np.std(scores)


def compute(font, trial, upper):
    file = f'{results_path}/trial_{trial}_font_{font}_{"upper" if upper else "lower"}.csv'
    log_file = f'{results_path}/trial_{trial}_font_{font}_{"upper" if upper else "lower"}.log'

    if os.path.exists(file):
        return
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    log = open(log_file, 'w')
    log.write('positives,negatives,weights\n')

    pos_neg = []
    scores = []
    errs = []

    uppers = np.array(list(map(lambda f: f[-9:-4] == 'upper', grams.graph_files)))

    positives_idx = shuffle(np.arange(len(grams.labels))[np.bitwise_and(grams.labels == font, uppers == upper)])
    negatives_idx = shuffle(np.arange(len(grams.labels))[np.bitwise_or(grams.labels != font, uppers != upper)])

    p_n = [(p, n) for p in [1, 2, 3, 4, 5, 10] for n in [0, 1, 2, 3, 4, 5, 50, 100]]
    for (p, n) in p_n:
        pos = positives_idx[:p]
        neg = negatives_idx[:n]

        weights = optimize(positive_idx=pos,
                           negative_idx=neg,
                           grams=reduced,
                           metric='cosine')
        log.write(f'{p},{n},"{weights.tolist()}"\n')
        weights = torch.tensor(weights).to(device)

        score, err = hits_at_k_score(reduced, weights, positives_idx, k=10)

        pos_neg.append((p, n))
        scores.append(score.mean())
        errs.append(err.mean())

    df = pd.DataFrame({
        'pos_neg': pos_neg,
        'score': scores,
        'err': errs
    })

    df.to_csv(file)
    log.close()


if __name__ == '__main__':
    device = torch.device('cuda:0')
    results_path = 'results_solidmnist_all_sub_mu'

    print('loading data...')
    grams = Grams('../../uvnet_data/solidmnist_all_sub_mu_only')
    reduced = pca_reduce(grams, 70, '../../cache/solidmnist_all_sub_mu_only')[:7]

    print('processing...')
    font_idx = {
        font_name: i for i, font_name in zip(grams.labels, map(lambda n: n[2:-10], grams.graph_files))
    }

    inputs = tqdm([
        (font, trial, upper)
        for trial in range(20)
        for font in [
            font_idx['Wire One'],
            font_idx['Viaoda Libre'],
            font_idx['Vast Shadow'],
            font_idx['Signika'],
            font_idx['Vampiro One'],
            font_idx['Stalemate'],
        ]
        for upper in [False]
    ])
    Parallel(-1)(delayed(compute)(*i) for i in inputs)
