import os
import sys

import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from abc_all.abc_top_k_by_layer import pad_grams

sys.path.append('../../../analysis')

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from all_fonts_svms.all_fonts_svm import pca_reduce
from constrained_optimization import optimize
from util import Grams


def hits_at_k_score(X, weights, positives, k=10):
    X = pad_grams(X).to(device)
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


def compute(font, trial, upper, l2):
    file = f'{results_path}/trial_{trial}_font_{font}_{"upper" if upper else "lower"}_l2_{l2}.csv'
    log_file = f'{results_path}/trial_{trial}_font_{font}_{"upper" if upper else "lower"}_l2_{l2}.log'

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

    p_n = [(p, n) for p in [1, 2, 3, 4, 5, 10] for n in [50]]
    for (p, n) in p_n:
        pos = positives_idx[:p]
        negatives_idx = list(set(np.arange(len(grams.labels))).difference(set(pos)))
        neg = shuffle(negatives_idx)[:n]

        weights = optimize(positive_idx=pos,
                           negative_idx=neg,
                           grams=reduced,
                           metric='cosine',
                           l2=l2)
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
    results_path = 'results_solidmnist_all_sub_mu_random_negatives_l2'

    print('loading data...')
    grams = Grams('../../uvnet_data/solidmnist_all_sub_mu')
    reduced = pca_reduce(grams, 70, '../../cache/solidmnist_all_sub_mu')[:7]

    print('processing...')
    font_idx = {
        font_name: i for i, font_name in zip(grams.labels, map(lambda n: n[2:-10], grams.graph_files))
    }

    inputs = tqdm([
        (font, trial, upper, l2)
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
        for l2 in [1e-6, 1e-5, 1e-3, 1e-2]
    ])
    Parallel(6)(delayed(compute)(*i) for i in inputs)
