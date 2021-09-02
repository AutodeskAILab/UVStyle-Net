import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from layer_stats_analysis import probe_score

if __name__ == '__main__':
    data_root = '../../uvnet_data/with_embeddings'
    content_embedding = np.load(f'{data_root}/content_embeddings.npy')
    graph_files = pd.read_csv(f'{data_root}/graph_files.txt', header=None, sep=',')
    graph_files['font'] = graph_files[0].apply(lambda f: f[2:-10])
    labels = graph_files['font'].astype('category').cat.codes.values

    print(content_embedding.shape)

    dims = []
    scores = []
    errs = []
    for dim in [None, 70, 25, 10, 3]:
        def reduce(x, dims):
            if dims is None or x.shape[-1] < dims:
                return x
            else:
                return PCA(n_components=dims).fit_transform(x)


        score, err = probe_score(dim, reduce(content_embedding, dim), labels, err=True)
        scores.append(score)
        errs.append(err)
        dims.append(dim)
    print('done probing...')

    df = pd.DataFrame({
        'dims': dims,
        'layer': 7,
        'score': scores,
        'err': errs
    })

    df.to_csv('dimension_reduction_probe_results_content.csv', index=False)
