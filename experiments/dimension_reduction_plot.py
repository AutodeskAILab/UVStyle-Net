import os
import sys
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(file_dir)
sys.path.append(project_root)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--include_content', action='store_true', default=False,
                        help='include content embeddings in plot (default: False)')
    args = parser.parse_args()
    df = pd.read_csv(os.path.join(file_dir, 'dimension_reduction_probe_results.csv'), sep=',')
    original_dims = [21, 2_080, 8_256, 32_896, 32_896, 2_080, 2_080, 136]
    layer_names = [
        '0_feats',
        '1_conv1',
        '2_conv2',
        '3_conv3',
        '4_fc',
        '5_GIN_1',
        '6_GIN_2',
        'content',
    ]
    legend_names = [
        '0_feats (21)',
        '1_conv1 (2,080)',
        '2_conv2 (8,256)',
        '3_conv3 (32,896)',
        '4_fc (3,2896)',
        '5_GIN1 (2,080)',
        '6_GIN2 (2,080)',
        'UV-Net Embedding\n(136)'
    ]

    fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
    xticks = np.arange(df['dims'].nunique())
    for i, (layer, group_df) in enumerate(df.groupby(['layer'])):
        group_df['dims'].replace(np.nan, original_dims[layer], inplace=True)
        if layer == 0:
            group_df = group_df.reset_index().drop([1, 2])
        elif not args.include_content and layer == 7:
            continue
        lines = ax.plot(group_df['dims'],
                        group_df['score'],
                        label=legend_names[layer])
        ax.plot(group_df['dims'].iloc[0],
                group_df['score'].iloc[0],
                'o',
                color=lines[-1].get_color())
        ax.fill_between(group_df['dims'],
                        group_df['score'] - group_df['err'],
                        group_df['score'] + group_df['err'],
                        alpha=0.2)

    ax.legend(fontsize='small')
    ax.set_xlabel('Reduced Dimensions')
    lim = ax.get_xlim()
    ax.semilogx()

    ax.plot([3, 32_896], [1 / 378, 1 / 378], '--', color='black')
    fig.set_size_inches(6, 3)
    fig.tight_layout()
    fig.savefig(os.path.join(file_dir, 'dimension_reduction_plot.pdf'))
    fig.show()
