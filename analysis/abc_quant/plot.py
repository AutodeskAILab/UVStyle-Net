from ast import literal_eval
from pathlib import Path

import pandas as pd

def to_cats(cats_list):
    cats = [Path(file).stem for file in cats_list]
    out = ' v '.join(cats)
    if cats_list[0].split('/')[1] == 'wheel_v_cog':
        out += ' (better)'
    return out

def to_model(data_root):
    return data_root.split('/')[1].split('_')[0]


if __name__ == '__main__':
    df = pd.read_csv('results_log_reg_cross_val_sklearn_balanced_f1_weighted.csv', sep=';')
    config_df = pd.DataFrame(df['config'].apply(literal_eval).values.tolist())
    df = pd.concat([df.drop('config', axis=1), config_df], axis=1)
    df['cats'] = df['cats_dirs'].apply(to_cats)
    df['model'] = df['data_root'].apply(to_model)
    uvnet = df.loc[df['model'] == 'uvnet']
    psnet = df.loc[df['model'] == 'psnet']

    df2 = uvnet.join(psnet.set_index('cats'), on='cats', how='left', lsuffix='_uvnet', rsuffix='_psnet')
    result = df2[['cats', 'mean_f1_uvnet', 'std_f1_uvnet', 'mean_f1_psnet', 'std_f1_psnet']]
    result['diff'] = result['mean_f1_uvnet'] - result['mean_f1_psnet']
    result.sort_values('diff', ascending=False, inplace=True)

    print(result[['cats', 'mean_f1_uvnet', 'std_f1_uvnet', 'mean_f1_psnet', 'std_f1_psnet','diff']].to_markdown())

    result['UVStyle-Net'] = result[['mean_f1_uvnet', 'std_f1_uvnet']].apply(lambda x: f'${x[0]:.3f} \pm {x[1]:.3f}$', axis=1)
    result['PSNet*'] = result[['mean_f1_psnet', 'std_f1_psnet']].apply(lambda x: f'${x[0]:.3f} \pm {x[1]:.3f}$', axis=1)

    result['UVStyle-Net'].loc[result['diff'] > 0] = result['UVStyle-Net'].loc[result['diff'] > 0].apply(lambda x: '$\mathbf{' + x[1:-1] + '}$')
    result['PSNet*'].loc[result['diff'] < 0] = result['PSNet*'].loc[result['diff'] < 0].apply(lambda x: '$\mathbf{' + x[1:-1] + '}$')

    print(result[['cats','UVStyle-Net','PSNet*','diff']].to_latex(escape=False))
