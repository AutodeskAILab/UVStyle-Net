from sklearn.metrics import silhouette_score
import pandas as pd

from layer_probing import knn_score
from layer_stats_analysis import probe_score
from util import Grams

if __name__ == '__main__':
    grams = Grams('../../uvnet_data/solidmnist_subset_no_inorm')
    df = pd.DataFrame(grams.layer_names, columns=['layer'])
    labels = grams.labels
    df['sil_euc'] = list(map(lambda x: silhouette_score(x, labels, metric='euclidean'), grams))
    df['sil_cos'] = list(map(lambda x: silhouette_score(x, labels, metric='cosine'), grams))
    df['knn_euc'] = list(map(lambda x: knn_score(x, labels, metric='euclidean'), grams))
    df['knn_cos'] = list(map(lambda x: knn_score(x, labels, metric='cosine'), grams))
    df['linear_probe'], df['linear_probe_err'] = zip(*list(map(lambda x: probe_score(None, x, labels, err=True), grams)))
    df.to_csv('uvnet_no_inorm_layer_probe_scores_with_err.csv', index=False)