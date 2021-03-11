import os
import subprocess
import sys

if __name__ == '__main__':

    image_cats = [f'abc_quant_data/{file}' for file in os.listdir('abc_quant_data/image_cats')]
    cross_cats = [['abc_quant_data/wheel_v_cog/wheel', 'abc_quant_data/wheel_v_cog/cog']]
    for i, cat_i in enumerate(image_cats):
        for j, cat_j in enumerate(image_cats):
            if j > i:
                cross_cats.append([cat_i, cat_j])

    print(cross_cats)

    NUM_TRIALS = 10
    configs = [
        {
            'data_root': data_root,
            'cats_dirs': cats_dirs,
            'l2': l2,
            'trial': trial
        }
        for data_root in ['../uvnet_data/abc_sub_mu_only', '../psnet_data/abc_all']
        for cats_dirs in cross_cats
        for l2 in ['0.1', '0.01', '0.001', '0.0001', '0.']
        for trial in list(range(NUM_TRIALS))
    ]

    for version, config in enumerate(configs):
        print(f'CONFIG: {version}/{len(configs)}')
        print(config)
        result = subprocess.run([sys.executable, 'logistic_regression.py',
                                 '--data_root', config['data_root'],
                                 '--cats_dir', *config['cats_dirs'],
                                 '--l2', config['l2'],
                                 '--version', str(version),
                                 '--trial', str(config['trial'])],
                                stdout=sys.stdout,
                                stderr=sys.stderr
                                )

