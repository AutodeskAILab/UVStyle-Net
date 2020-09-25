import pandas as pd
from sklearn.decomposition import PCA
import sys
sys.path.append('../../../analysis')
from layer_stats_analysis import probe_score
from util import Grams

if __name__ == '__main__':
    grams = Grams('../../uvnet_data/solidmnist_all')
    df = pd.DataFrame(grams.layer_names, columns=['layer'])
    labels = grams.labels
    df['linear_probe'], df['linear_probe_err'] = zip(*list(map(lambda x: probe_score(None, PCA(70).fit_transform(x) if x.shape[-1] > 70 else x, labels, err=True), grams)))
    df.to_csv('uvnet_layer_probe_scores_all_with_err.csv', index=False)