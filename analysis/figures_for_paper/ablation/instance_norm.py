import pandas as pd
import sys
sys.path.append('../../../analysis/')

from layer_stats_analysis import probe_score
from util import Grams

if __name__ == '__main__':
    versions = [
        # 'solidmnist_all_raw_grams',
        # 'solidmnist_all_inorm',
        # 'solidmnist_all_fnorm',
        # 'solidmnist_subset_feats_mu_sigma',
        # 'solidmnist_subset_feats_mu_sigma_inorm',
        # 'solidmnist_subset_feats_mu',
        # 'solidmnist_subset_feats_mu_inorm',
        # 'solidmnist_subset_mu_sigma_inorm_concat_fnorm'
        'solidmnist_subset_feats_sigma',
        'solidmnist_subset_feats_sigma_inorm',
    ]
    for version in versions:
        print(f'running {version}...')
        grams = Grams(f'../../uvnet_data/{version}')
        df = pd.DataFrame(grams.layer_names, columns=['layer'])
        labels = grams.labels
        df['linear_probe'], df['linear_probe_err'] = zip(*list(map(lambda x: probe_score(None, x, labels, err=True), grams)))
        df.to_csv(f'uvnet_{version}_layer_probe_scores_with_err.csv', index=False)