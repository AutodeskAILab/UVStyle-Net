import os
import sys

file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(file_dir)
sys.path.append(project_root)
from pytorch_probe_score import probe_score
from utils import Grams

if __name__ == '__main__':
    versions = [
        'all_raw_grams',
        'all_inorm_only',
        'all_fnorm_only',
        'all'
    ]
    for version in versions:
        print(f'running {version}...')
        grams = Grams(os.path.join(project_root, 'data', 'SolidLETTERS', 'grams', version))
        df = probe_score(grams, batch_size=128, fast_dev_run=False)
        df.to_csv(os.path.join(file_dir, f'uvnet_{version}_layer_probe_scores_with_err.csv'), index=False)
