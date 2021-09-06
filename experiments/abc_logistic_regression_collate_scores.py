import os
import sys
from ast import literal_eval
from pathlib import Path

import pandas as pd

file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(file_dir)
sys.path.append(project_root)

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(file_dir, 'abc_quant_results.csv'), sep=';')
    config_df = pd.DataFrame(df['config'].apply(literal_eval).values.tolist())
    df = pd.concat([df.drop('config', axis=1), config_df], axis=1)


    def to_cats(cats_list):
        cats = [Path(file).stem for file in cats_list]
        out = ' v '.join(cats)
        return out


    def to_model(data_root):
        return data_root.split('/')[7].split('_')[0]


    df['cat'] = df['cats_dirs'].apply(to_cats)
    df['model'] = df['data_root'].apply(to_model)

    df = df.drop(['cats_dirs', 'data_root'], axis=1)

    mean_over_trials = df.groupby(['model', 'cat', 'l2']).mean().reset_index()
    std_over_trials = df.groupby(['model', 'cat', 'l2']).std().reset_index()[['test_acc']]

    mean_over_trials['test_std'] = std_over_trials

    best_val_idx = mean_over_trials.groupby(['model', 'cat']).idxmax()['val_acc']

    best = mean_over_trials.loc[best_val_idx]

    results = []
    for cat, group_df in best.groupby(['cat']):
        try:
            uvnet_acc = group_df.loc[group_df['model'] == 'uvnet']['test_acc'].__float__()
            uvnet_std = group_df.loc[group_df['model'] == 'uvnet']['test_std'].__float__()
        except:
            uvnet_acc, uvnet_std = 0., 0.
        try:
            psnet_acc = group_df.loc[group_df['model'] == 'psnet']['test_acc'].__float__()
            psnet_std = group_df.loc[group_df['model'] == 'psnet']['test_std'].__float__()
        except:
            psnet_acc, psnet_std = 0., 0.

        result = {
            'Category': cat,
            'UVStyle-Net': f'{uvnet_acc:.3f} +/- {uvnet_std:.3f}',
            'PSNet': f'{psnet_acc:.3f} +/- {psnet_std:.3f}',
            'Diff': uvnet_acc - psnet_acc
        }

        if uvnet_acc > psnet_acc:
            result['UVStyle-Net'] = f'**{result["UVStyle-Net"]}**'
        else:
            result['PSNet'] = f'**{result["PSNet"]}**'

        results.append(result)

    final_df = pd.DataFrame(results).sort_values('Diff', ascending=False).reset_index()
    print(final_df.to_markdown().replace('_', '\_'))
