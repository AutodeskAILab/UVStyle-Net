import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import sys

sys.path.append('../../../analysis/')
# from layer_stats_analysis import probe_score
from pytorch_probe_score_2 import probe_score
from util import Grams

# scores, errs = zip(*list(map(lambda x: probe_score(None, reduce(x, dims), labels, err=True), grams)))

if __name__ == '__main__':
    #grams = Grams('../../uvnet_data/solidmnist_all_fnorm')
    grams = Grams('../../uvnet_data/solidmnist_all_sub_mu_only')
    labels = grams.labels
    dfs = []
    print(grams)
    #print(grams.graph_files.shape)
    for g in grams:
        print(g.shape)
    for dims in [None, 70, 25, 10, 3]:
        print('DIMS', dims)
        def reduce(x, dims):
            if dims is None or x.shape[-1] < dims:
                return x
            else:
                return PCA(n_components=dims).fit_transform(x)

        reduced = list(map(lambda x: reduce(x, dims), grams))
        print(len(reduced))
        scores = probe_score(reduced, labels, grams.layer_names)
        # scores, errs = zip(*list(map(lambda x, y: probe_score(reduce(x), y), grams,)))                
        # dfs.append(pd.DataFrame({ 'dims': dims, 'layer': np.arange(len(scores)),'score': scores, 'err': errs}))
        print('another size', scores)
        #dfs.append(pd.DataFrame({ 'dims': dims, 'layer': np.arange(len(scores)),'score': scores}))


    #df = pd.concat(dfs, axis=0)
        scores.to_csv(f'dimension_reduction_probe_results_{dims}.csv', index=False)
