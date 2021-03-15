import subprocess
import sys

from joblib import Parallel, delayed
from tqdm import tqdm


def compute(version, config):
    # print(config)
    subprocess.run([sys.executable, 'svm.py',
                    '--log', log_file,
                    '--data_root', config['data_root'],
                    '--cats_dir', *config['cats_dirs'],
                    '--version', str(version),
                    '--trial', str(config['trial'])],
                   # stdout=sys.stdout,
                   # stderr=sys.stderr
                   )


if __name__ == '__main__':

    log_file = 'svm_results_all_poly_all.csv'

    data_roots = ['../uvnet_data/abc_sub_mu_only', '../psnet_data/abc_all']
    # image_cats = [f'abc_quant_data/image_cats/{file}' for file in os.listdir('abc_quant_data/image_cats')]
    cross_cats = [['abc_quant_data/wheel_v_cog/wheel', 'abc_quant_data/wheel_v_cog/cog']]
    # cross_cats = []
    # for i, cat_i in enumerate(image_cats):
    #     for j, cat_j in enumerate(image_cats):
    #         if j > i:
    #             cross_cats.append([cat_i, cat_j])

    print(cross_cats)

    NUM_TRIALS = 1
    configs = [
        {
            'data_root': data_root,
            'cats_dirs': cats_dirs,
            'trial': trial,
        }
        for trial in list(range(NUM_TRIALS))
        for cats_dirs in cross_cats
        for data_root in data_roots
    ]

    inputs = tqdm(enumerate(configs))
    Parallel(1)(delayed(compute)(*i) for i in inputs)
