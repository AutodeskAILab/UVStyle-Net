import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from layer_stats_analysis import probe_score
from util import Grams

if __name__ == '__main__':
    grams = Grams('../../uvnet_data/solidmnist_font_subset')
    labels = grams.labels
    dfs = []
    for dims in [None, 70, 25, 10, 3]:
        def reduce(x, dims):
            if dims is None or x.shape[-1] < dims:
                return x
            else:
                return PCA(n_components=dims).fit_transform(x)


        scores, errs = zip(*list(map(lambda x: probe_score(None, reduce(x, dims), labels, err=True), grams)))
        dfs.append(pd.DataFrame({
            'dims': dims,
            'layer': np.arange(len(scores)),
            'score': scores,
            'err': errs
        }))

    df = pd.concat(dfs, axis=0)
    df.to_csv('dimension_reduction_probe_results.csv', index=False)
