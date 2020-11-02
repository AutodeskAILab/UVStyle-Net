import sys

sys.path.append('../../../analysis/')
from pytorch_probe_score import probe_score
from util import Grams

if __name__ == '__main__':
    versions = [
        # 'solidmnist_all_raw_grams',
        # 'solidmnist_all_inorm',
        # 'solidmnist_all_fnorm',
        'solidmnist_all_sub_mu_only'
    ]
    for version in versions:
        print(f'running {version}...')
        grams = Grams(f'../../uvnet_data/{version}')
        df = probe_score(grams, batch_size=128, fast_dev_run=False)
        df.to_csv(f'uvnet_{version}_layer_probe_scores_with_err.csv', index=False)
