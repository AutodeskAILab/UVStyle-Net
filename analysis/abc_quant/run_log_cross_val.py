from argparse import Namespace

from abc_quant.logistic_cross_val import main


def compute(version, config):
    # print(config)

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
    config['log'] = 'results_log_reg_cross_val_sklearn_balanced_f1_weighted.csv'
    args = Namespace(**config)
    main(args)


if __name__ == '__main__':
    data_roots = ['../uvnet_data/abc_sub_mu_only', '../psnet_data/abc_all']
    cross_cats = [
        ['abc_quant_data/image_cats/cog', 'abc_quant_data/image_cats/wheel'],
        ['abc_quant_data/image_cats/free_form', 'abc_quant_data/image_cats/pipe'],
        ['abc_quant_data/image_cats/angular', 'abc_quant_data/image_cats/rounded'],
        ['abc_quant_data/image_cats/flat', 'abc_quant_data/image_cats/electric'],
    ]

    # image_cats = [f'abc_quant_data/image_cats/{file}' for file in os.listdir('abc_quant_data/image_cats')]
    # cross_cats = [['abc_quant_data/wheel_v_cog/wheel', 'abc_quant_data/wheel_v_cog/cog']]
    # # cross_cats = []
    # for i, cat_i in enumerate(image_cats):
    #     for j, cat_j in enumerate(image_cats):
    #         if j > i:
    #             cross_cats.append([cat_i, cat_j])

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
