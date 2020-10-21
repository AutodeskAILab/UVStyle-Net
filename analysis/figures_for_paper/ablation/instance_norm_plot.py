import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':

    dfs = {
        'Gram': pd.read_csv('uvnet_solidmnist_all_raw_grams_layer_probe_scores_with_err.csv'),
        'INorm Only': pd.read_csv('uvnet_solidmnist_all_inorm_layer_probe_scores_with_err.csv'),
        'FNorm Only': pd.read_csv('uvnet_solidmnist_all_fnorm_layer_probe_scores_with_err.csv'),
    }

    for i, (version, df) in enumerate(dfs.items()):
        if version in ['FNorm Only', 'FNorm + cat mu & sigma']:
            xticks = np.arange(4) * (len(dfs) + 1)
            df = df.loc[:3]
        else:
            xticks = np.arange(len(df)) * (len(dfs) + 1)
        plt.bar(x=xticks + i,
                height=df['linear_probe'],
                yerr=df['linear_probe_err'],
                label=version)

    plt.legend()
    ax = plt.gca()  # type: plt.Axes
    labels = [
        '0_feats',
        '1_conv',
        '2_conv',
        '3_conv',
        '4_fc',
        '5_GIN',
        '6_GIN'
    ]
    xticks = np.arange(len(labels) + 1) * (len(dfs)+1) + 0.5
    ax.set_xticks(xticks[:-1] + 1)
    ax.set_xticklabels(labels)
    plt.ylim([.8, 1.])
    plt.tight_layout()
    plt.savefig('inorm_plot.pdf')
    plt.show()
