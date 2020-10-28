import os
import sys

from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append('../../../analysis')
from all_fonts_svms.all_fonts_weight_layers_gaussian import hits_at_k_score

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from all_fonts_svms.all_fonts_svm import pca_reduce
from constrained_optimization import optimize
from util import Grams, weight_layers


def compute(font, trial, upper):
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
        combined = weight_layers(reduced, weights)

        score, err = hits_at_k_score(combined, grams.labels, font, k=10)

        pos_neg.append((p, n))
        scores.append(score.mean())
        errs.append(err.mean())

    df = pd.DataFrame({
        'pos_neg': pos_neg,
        'score': scores,
        'err': errs
    })

    if not os.path.exists(results_path):
        os.mkdir(results_path)

    df.to_csv(f'{results_path}/trial_{trial}_font_{font}_{"upper" if upper else "lower"}.csv')


if __name__ == '__main__':
    results_path = 'results_solidmnist_all_fnorm_by_case'

    print('loading data...')
    grams = Grams('../../uvnet_data/solidmnist_all_fnorm')
    reduced = pca_reduce(grams, 70, '../../cache/solidmnist_all_fnorm')

    font_idx = {
        font_name: i for i, font_name in zip(grams.labels, map(lambda n: n[2:-10], grams.graph_files))
    }

    inputs = tqdm([
        (font, trial, upper)
        for font in [font_idx['Wire One']]
        # for font in [font_idx['Viaoda Libre'], font_idx['Vast Shadow']]
        for trial in range(20, 50)
        for upper in [True, False]
    ])
    Parallel(-1)(delayed(compute)(*i) for i in inputs)
