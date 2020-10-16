import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':

    dfs = {
        'Without INorm': pd.read_csv('uvnet_solidmnist_font_subset_no_inorm_layer_probe_scores_with_err.csv'),
        'With INorm': pd.read_csv('uvnet_solidmnist_font_subset_layer_probe_scores_with_err.csv'),
        'New Grams': pd.read_csv('uvnet_solidmnist_font_subset_new_grams_layer_probe_scores_with_err.csv')
    }

    for i, (version, df) in enumerate(dfs.items()):
        xticks = np.arange(len(df)) * 4
        plt.bar(x=xticks + i,
                height=df['linear_probe'],
                yerr=df['linear_probe_err'],
                label=version)

    plt.legend()
    ax = plt.gca()  # type: plt.Axes
    ax.set_xticks(xticks[:-1] + 1)
    ax.set_xticklabels(labels=[
        '0_feats',
        '1_conv',
        '2_conv',
        '3_conv',
        '4_fc',
        '5_GIN',
        '6_GIN'
    ])
    plt.ylim([.8, 1.])
    plt.tight_layout()
    plt.savefig('inorm_plot.pdf')
    plt.show()
