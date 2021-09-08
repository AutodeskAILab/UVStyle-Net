import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(file_dir)
sys.path.append(project_root)
from utils import Grams, probe_score


def reduce(x, dims):
    if dims is None or x.shape[-1] < dims:
        return x
    else:
        return PCA(n_components=dims).fit_transform(x)


if __name__ == '__main__':
    grams = Grams(os.path.join(project_root, 'data', 'SolidLETTERS', 'grams', 'all'))
    labels = grams.labels
    dfs = []
    for dims in [None, 70, 25, 10, 3]:
        print(f'Dimensions: {dims or "Original"}')
        scores, errs = zip(*[probe_score(i, reduce(x, dims), labels, err=True) for i, x in enumerate(grams)])
        dfs.append(pd.DataFrame({
            'dims': dims,
            'layer': np.arange(len(scores)),
            'score': scores,
            'err': errs
        }))

    df = pd.concat(dfs, axis=0)
    df.to_csv(os.path.join(file_dir, 'dimension_reduction_probe_results.csv'), index=False)
