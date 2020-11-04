import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_df(d):
    df = pd.read_csv(f'dim_reduction_results/dimension_reduction_probe_results_{d}.csv', sep=',')
    df['dims'] = [21, 2_080, 8_256, 32_896, 32_896, 2_080, 2_080, 136] if d is None else d
    return df

if __name__ == '__main__':
    df = pd.concat([load_df(d) for d in [None, 70, 25, 10, 3]])
    layer_names = {
        '0_feats': '0_feats (21)',
        '1_conv1': '1_conv (2,080)',
        '2_conv2': '2_conv (8,256)',
        '3_conv3': '3_conv (32,896)',
        '4_fc': '4_fc (3,2896)',
        '5_GIN_1': '5_GIN (2,080)',
        '6_GIN_2': '6_GIN (2,080)',
        'content': 'UV-Net Embedding\n(136)'
    }

    fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
    xticks = np.arange(df['dims'].nunique())
    for i, (layer, group_df) in enumerate(df.groupby(['layer'])):
        if layer == '0_feats':
            group_df = group_df.reset_index().drop([1, 2])
        elif layer == 'content':
            continue
        ax.plot(group_df['dims'],
                group_df['linear_probe'],
                label=layer_names[layer])
        ax.fill_between(group_df['dims'],
                        group_df['linear_probe'] - group_df['linear_probe_err'],
                        group_df['linear_probe'] + group_df['linear_probe_err'],
                        alpha=0.2)

    ax.legend()
    ax.set_xlabel('Reduced Dimensions')
    lim = ax.get_xlim()
    # ax.loglog()
    ax.semilogx()

    ax.plot([3, 32_896 * 1.2], [1/378, 1/378], '--', color='black')
    fig.set_size_inches(6, 3)
    fig.tight_layout()
    fig.savefig('dimension_reduction_plot.pdf')
    fig.show()
