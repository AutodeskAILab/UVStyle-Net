import os

import numpy as np
import pandas as pd
from progressbar import progressbar
from sklearn.neighbors._kd_tree import KDTree

from all_fonts_svms.all_fonts_svm import pca_reduce
from util import Grams, weight_layers


def spherize(emb):
    norm = np.linalg.norm(emb, axis=-1, keepdims=True)
    return emb / norm


def hits_at_k_score(X, y, target, k=10, metric='cosine'):
    if metric == 'cosine':
        X = spherize(X)
    kd_tree = KDTree(X)
    Q = X[y == target]
    results = kd_tree.query(Q, k=k + 1, return_distance=False)[:, 1:]
    result_classes = y[results]
    scores = []
    for i in range(len(result_classes)):
        score = sum(result_classes[i] == target) / k
        # print(result_classes[i], score)
        scores.append(score)
    return np.mean(scores), np.std(scores)


def pdf(x, mu, sigma):
    A = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = - .5 * ((x - mu) / sigma) ** 2
    return A * np.exp(exponent)


def weight_layers_gaussian(X, mu, sigma):
    weights = pdf(np.arange(len(X)), mu, sigma)
    # print(f'\nmu: {mu}, sigma: {sigma}, weights: {weights}')
    combined = np.concatenate([weights[i] * X[i] for i in range(len(X))], axis=-1)
    return combined


if __name__ == '__main__':
    results_path = 'gaussian_weights_scores'

    grams = Grams('../uvnet_data/solidmnist_filtered_font_families')
    reduced = pca_reduce(grams, 18, '../cache/solidmnist_filtered_font_families')
    y = grams.labels

    mus = np.linspace(start=0., stop=6., num=13)
    sigmas = [0.05, 0.1, 0.25, 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.]
    mus_sigmas = [(mu, sigma) for sigma in sigmas for mu in mus]

    for font in range(len(set(y))):
        scores = []
        errs = []
        for mu, sigma in progressbar(mus_sigmas, prefix=f'font: {font}'):
            X = weight_layers_gaussian(X=reduced, mu=mu, sigma=sigma)

            score, err = hits_at_k_score(X, y, target=font, k=10, metric='cosine')
            scores.append(score)
            errs.append(err)
            # print(f'\nmu: {mu}, sigma: {sigma}, score: {score:.5f} +/- {err:.5f}')

        df = pd.DataFrame(data={
            'mu_sigma': mus_sigmas,
            'score': scores,
            'err': errs
        })

        print(df.max())

        if not os.path.exists(results_path):
            os.mkdir(results_path)

        df.to_csv(f'{results_path}/font_{font}.csv', sep=',', index=False)
