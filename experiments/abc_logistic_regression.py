import subprocess
import sys
import os
from argparse import ArgumentParser

from joblib import Parallel, delayed
from tqdm import tqdm

file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(file_dir)
sys.path.append(project_root)


def compute(version, config):
    # print(config)
    subprocess.run([sys.executable, os.path.join(file_dir, 'logistic_regression.py'),
                    '--data_root', config['data_root'],
                    '--cats_dir', *config['cats_dirs'],
                    '--l2', config['l2'],
                    '--version', str(version),
                    '--trial', str(config['trial'])],
                   # stdout=sys.stdout,
                   # stderr=sys.stderr
                   )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_threads', default=5, type=int,
                        help='number of concurrent threads (default: 5)')
    args = parser.parse_args()

    # todo: sort out data paths
    data_roots = [os.path.join(project_root, 'analysis/uvnet_data/abc_sub_mu_only'),
                  os.path.join(project_root, 'analysis/psnet_data/abc_all')]
    cross_cats = [
        [os.path.join(project_root, 'analysis/abc_quant/abc_quant_data/image_cats/flat'),
         os.path.join(project_root, 'analysis/abc_quant/abc_quant_data/image_cats/electric')],
        [os.path.join(project_root, 'analysis/abc_quant/abc_quant_data/image_cats/free_form'),
         os.path.join(project_root, 'analysis/abc_quant/abc_quant_data/image_cats/pipe')],
        [os.path.join(project_root, 'analysis/abc_quant/abc_quant_data/image_cats/angular'),
         os.path.join(project_root, 'analysis/abc_quant/abc_quant_data/image_cats/rounded')],
    ]

    NUM_TRIALS = 2
    configs = [
        {
            'data_root': data_root,
            'cats_dirs': cats_dirs,
            'l2': l2,
            'trial': trial
        }
        for trial in list(range(NUM_TRIALS))
        for cats_dirs in cross_cats
        for data_root in data_roots
        for l2 in ['0.1', '0.001', '0.']
    ]

    inputs = tqdm(enumerate(configs))
    Parallel(args.num_threads)(delayed(compute)(*i) for i in inputs)
