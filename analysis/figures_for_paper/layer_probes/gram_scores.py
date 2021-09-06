import sys

import pandas as pd
from sklearn.decomposition import PCA

sys.path.append('../../../analysis')
from layer_stats_analysis import probe_score
from util import Grams

if __name__ == '__main__':
    grams = Grams('../../uvnet_data/solidmnist_font_subset')
    df = pd.DataFrame(grams.layer_names, columns=['layer'])
    labels = grams.labels
    df['linear_probe'], df['linear_probe_err'] = zip(
        *[probe_score(i, PCA(70).fit_transform(x) if x.shape[-1] > 70 else x, labels, err=True) for i, x in
          enumerate(grams)])
    df.to_csv('test.csv', index=False)
