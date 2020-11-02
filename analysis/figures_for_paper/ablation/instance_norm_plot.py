import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':

    dfs = {
        'Gram': pd.read_csv('uvnet_solidmnist_all_raw_grams_layer_probe_scores_with_err_l2.csv'),
        'INorm Only': pd.read_csv('uvnet_solidmnist_all_inorm_layer_probe_scores_with_err_l2.csv'),
        'FNorm Only': pd.read_csv('uvnet_solidmnist_all_fnorm_layer_probe_scores_with_err_l2.csv'),
        'Sub Mu Only': pd.read_csv('uvnet_solidmnist_all_sub_mu_only_layer_probe_scores_with_err.csv'),
    }

    fig, ax = plt.subplots() # type: plt.Figure, plt.Axes
    for i, (version, df) in enumerate(dfs.items()):
        if version == 'FNorm Only':
            df = df.loc[0:3]
        else:
            df = df.loc[0:6]
        xticks = np.arange(len(df)) * (len(dfs) + 1)
        ax.bar(x=xticks + i,
                height=df['linear_probe'],
                yerr=df['linear_probe_err'],
                label=version)

    ax.legend()
    labels = [
        '0_feats',
        '1_conv',
        '2_conv',
        '3_conv',
        '4_fc',
        '5_GIN',
        '6_GIN',
    ]
    xticks = np.arange(len(labels)) * (len(dfs)+1)
    ax.set_xticks(xticks + 1.5)
    ax.set_xticklabels(labels)
    # random baseline
    baseline = 1 / 378
    ax.plot([-1, (len(labels))*(len(dfs) + 1) - 1], [baseline, baseline], '--', color='black')
    # plt.ylim([.8, 1.])
    fig.set_size_inches(5, 2.5)
    fig.tight_layout()
    fig.savefig('inorm_plot.pdf')
    fig.show()
