import pandas as pd

from layer_stats_analysis import probe_score
from util import Grams

if __name__ == '__main__':
    versions = [
        'solidmnist_font_subset_no_inorm',
        'solidmnist_font_subset',
        'solidmnist_font_subset_new_grams'
    ]
    for version in versions:
        print(f'running {version}...')
        grams = Grams(f'../../uvnet_data/{version}')
        df = pd.DataFrame(grams.layer_names, columns=['layer'])
        labels = grams.labels
        df['linear_probe'], df['linear_probe_err'] = zip(*list(map(lambda x: probe_score(None, x, labels, err=True), grams)))
        df.to_csv(f'uvnet_{version}_layer_probe_scores_with_err.csv', index=False)