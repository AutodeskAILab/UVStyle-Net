import os
import sys
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.utils import shuffle
from tqdm import tqdm
from itertools import islice

file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(file_dir)
sys.path.append(project_root)
from utils import Grams, pad_grams, get_pca_70
from few_shot import optimize


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


def compute(font, trial, upper):
    file = os.path.join(args.exp_name, f'trial_{trial}_font_{font}_{"upper" if upper else "lower"}.csv')
    log_file = os.path.join(args.exp_name, f'trial_{trial}_font_{font}_{"upper" if upper else "lower"}.log')

    if os.path.exists(file):
        return
    if not os.path.exists(args.exp_name):
        os.makedirs(args.exp_name)

    log = open(log_file, 'w')
    log.write('positives,negatives,weights\n')

    pos_neg = []
    scores = []
    errs = []

    uppers = np.array(list(map(lambda f: f[-9:-4] == 'upper', grams.graph_files)))

    positives_idx = shuffle(np.arange(len(grams.labels))[np.bitwise_and(grams.labels == font, uppers == upper)])

    p_n = [(p, n) for p in [1, 2, 3, 4, 5, 10] for n in [0, 50, 100]]
    for (p, n) in p_n:
        pos = positives_idx[:p]
        negatives_idx = list(set(np.arange(len(grams.labels))).difference(set(pos)))
        neg = shuffle(negatives_idx)[:n]

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
    parser = ArgumentParser()
    parser.add_argument('--exp_name', default='SolidLETTERS-all', type=str,
                        help='experiment name - '
                             'results for each font/trial will be '
                             'saved into this directory '
                             '(default: SolidLETTERS-all)')
    parser.add_argument('--num_threads', default=6, type=int,
                        help='number of concurrent threads (default: 6)')
    args = parser.parse_args()

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    print('loading data...')
    grams = Grams(os.path.join(project_root, 'data', 'SolidLETTERS', 'grams', 'all'))
    os.makedirs(os.path.join(file_dir, 'cache'), exist_ok=True)
    reduced_od = get_pca_70(grams, os.path.join(file_dir, 'cache', f'{args.exp_name}-pca_70'))
    reduced = list(islice(reduced_od.values(), 7))

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
    Parallel(args.num_threads)(delayed(compute)(*i) for i in inputs)
