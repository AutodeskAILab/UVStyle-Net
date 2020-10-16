import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from util import Grams, weight_layers


def pca_reduce(X, d, cache_file):
    def fit_transform(arr, d):
        if arr.shape[-1] > d:
            return PCA(d).fit_transform(arr)
        else:
            return arr
    cache_file = cache_file + f'_{d}.pickle'
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as file:
            reduced = pickle.load(file)
    else:
        reduced = list(map(lambda x: fit_transform(x, d), X))
        with open(cache_file, 'wb') as file:
            pickle.dump(reduced, file)
    return reduced


if __name__ == '__main__':
    print('loading data...')
    grams = Grams('../uvnet_data/solidmnist_all')
    reduced = pca_reduce(grams, 18, '../cache/solidmnist_all')
    combined = weight_layers(reduced, np.ones(len(reduced)))

    print('computing kernel matrix...')
    kernel_matrix = cosine_distances(combined)

    print('fitting SVMs...')
    scores = {}
    errs = {}
    for clazz in tqdm(sorted(list(set(grams.labels)))):
        cls = SVC(kernel='precomputed', class_weight='balanced')
        score = cross_val_score(estimator=cls,
                                X=kernel_matrix,
                                y=grams.labels == clazz,
                                scoring='roc_auc',
                                n_jobs=-1)
        scores[clazz] = score.mean()
        errs[clazz] = score.std()

    df = pd.merge(left=pd.DataFrame.from_dict(scores, columns=['acc'], orient='index'),
                  right=pd.DataFrame.from_dict(errs, columns=['std'], orient='index'),
                  left_index=True,
                  right_index=True,
                  how='outer')
    df.to_csv('all_fonts_svm_results.csv', sep=',')
