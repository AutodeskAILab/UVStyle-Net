import os
import sys

from joblib import Parallel, delayed
from tqdm import tqdm

from abc_all.abc_top_k_by_layer import gram_loss

sys.path.append('../../../analysis')

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from all_fonts_svms.all_fonts_svm import pca_reduce
from constrained_optimization import optimize
from util import Grams


def hits_at_k_score(X, weights, y, target, k=10, metric='cosine'):
    queries = np.where(y == target)[0]
    scores = []
    for query in queries:
        distances = np.zeros(len(y))
        for other in range(len(y)):
            _, distances[other] = gram_loss(X, query, other, weights, metric=metric)
        results = np.argsort(distances)[:k]
        result_classes = y[results]
        score = sum(result_classes == target) / k
        scores.append(score)
    return np.mean(scores), np.std(scores)


def compute(font, trial, upper):
    file = f'{results_path}/trial_{trial}_font_{font}_{"upper" if upper else "lower"}.csv'
    if os.path.exists(file):
        return

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

        score, err = hits_at_k_score(reduced, weights, grams.labels, font, k=10)

        pos_neg.append((p, n))
        scores.append(score.mean())
        errs.append(err.mean())

    df = pd.DataFrame({
        'pos_neg': pos_neg,
        'score': scores,
        'err': errs
    })

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    df.to_csv(file)


if __name__ == '__main__':
    results_path = 'results_solidmnist_all_sub_mu'

    print('loading data...')
    grams = Grams('../../uvnet_data/solidmnist_all_sub_mu')
    reduced = pca_reduce(grams, 70, '../../cache/solidmnist_all_sub_mu')

    print('processing...')
    font_idx = {
        font_name: i for i, font_name in zip(grams.labels, map(lambda n: n[2:-10], grams.graph_files))
    }

    inputs = tqdm([
        (font, trial, upper)
        for font in [
            font_idx['Wire One'],
            font_idx['Viaoda Libre'],
            font_idx['Vast Shadow'],
            font_idx['Signika'],
            font_idx['Vampiro One'],
            font_idx['Stalemate'],
        ]
        for trial in range(20)
        for upper in [False]
    ])
    Parallel(1)(delayed(compute)(*i) for i in inputs)
