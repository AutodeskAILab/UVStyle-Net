import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':

    dfs = {
        'Gram': pd.read_csv('uvnet_solidmnist_all_raw_grams_layer_probe_scores_with_err_l2.csv'),
        'INorm': pd.read_csv('uvnet_solidmnist_all_inorm_layer_probe_scores_with_err_l2.csv'),
        # 'FNorm Only': pd.read_csv('uvnet_solidmnist_all_fnorm_layer_probe_scores_with_err_l2.csv'),
        'Face\nRe-centering': pd.read_csv('uvnet_solidmnist_all_sub_mu_only_layer_probe_scores_with_err.csv'),
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

    labels = [
        '0_feats',
        '1_conv1',
        '2_conv2',
        '3_conv3',
        '4_fc',
        '5_GIN1',
        '6_GIN2',
        'UV-Net'
    ]
    xticks = np.arange(len(labels) + 1) * (len(dfs)+1)
    ax.set_xticks(xticks + 1)
    ax.set_xticklabels(labels, fontsize='small')

    ax.bar(x=7*4+1,
           height=0.007656926661729813,
           yerr=0.0018307526143548228,
           label='UV-Net\nEmbedding')

    ax.legend(fontsize='small')


    # random baseline
    baseline = 1 / 378
    ax.plot([-1, (len(labels))*(len(dfs) + 1) - 1], [baseline, baseline], '--', color='black')
    plt.ylim([0., .3])
    fig.set_size_inches(5, 2)
    fig.tight_layout()
    fig.savefig('inorm_plot.pdf')
    fig.show()
