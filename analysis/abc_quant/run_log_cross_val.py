import os
import subprocess
import sys
from argparse import Namespace

from joblib import Parallel, delayed
from numpy.distutils.system_info import system_info
from tqdm import tqdm

from abc_quant.logistic_cross_val import main


def compute(version, config):
    # print(config)
    if version in [0,
                   1,
                   2,
                   3,
                   4,
                   5,
                   6,
                   7,
                   8,
                   9,
                   10,
                   11,
                   12,
                   13,
                   14,
                   15,
                   16,
                   17,
                   18,
                   19,
                   20,
                   21,
                   25,
                   27,
                   29,
                   31,
                   33,
                   34,
                   35,
                   36,
                   37,
                   38,
                   39,
                   40,
                   41,
                   42,
                   43,
                   44,
                   45,
                   57,
                   58,
                   59,
                   60,
                   61,
                   62,
                   63,
                   64,
                   65,
                   66,
                   67,
                   77,
                   78,
                   79,
                   80,
                   81,
                   82,
                   83,
                   84,
                   85,
                   86,
                   87,
                   91,
                   93,
                   95,
                   96,
                   97,
                   98,
                   99,
                   113,
                   115,
                   117,
                   119,
                   121,
                   129,
                   133, ]:
        return
    # subprocess.run([
    #     sys.executable, 'logistic_cross_val.py',
    #     '--data_root', config['data_root'],
    #     '--cats_dir', *config['cats_dirs'],
    #     '--version', str(version),
    #     '--trial', str(config['trial']),
    #     '--log', 'results_log_reg_cross_val_sklearn_balanced.csv'
    # ],
    #     stdout=sys.stdout,
    #     stderr=sys.stderr)

    config['version'] = version
    config['log'] = 'results_log_reg_cross_val_sklearn_balanced.csv'
    args = Namespace(**config)
    main(args)


if __name__ == '__main__':
    data_roots = ['../uvnet_data/abc_sub_mu_only', '../psnet_data/abc_all']
    # cross_cats = [
    #     ['abc_quant_data/image_cats/extrude', 'abc_quant_data/image_cats/rounded'],
    #     ['abc_quant_data/image_cats/pipe', 'abc_quant_data/image_cats/rounded'],
    #     ['abc_quant_data/image_cats/pipe', 'abc_quant_data/image_cats/extrude'],
    # ]

    image_cats = [f'abc_quant_data/image_cats/{file}' for file in os.listdir('abc_quant_data/image_cats')]
    cross_cats = [['abc_quant_data/wheel_v_cog/wheel', 'abc_quant_data/wheel_v_cog/cog']]
    # cross_cats = []
    for i, cat_i in enumerate(image_cats):
        for j, cat_j in enumerate(image_cats):
            if j > i:
                cross_cats.append([cat_i, cat_j])

    NUM_TRIALS = 1
    configs = [
        {
            'data_root': data_root,
            'cats_dirs': cats_dirs,
            'trial': 0,
        }
        for cats_dirs in cross_cats
        for data_root in data_roots
    ]

    # inputs = enumerate(configs)
    # Parallel(1)(delayed(compute)(*i) for i in inputs)
    for i, input in enumerate(configs):
        compute(i, input)
