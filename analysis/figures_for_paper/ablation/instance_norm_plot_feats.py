import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':

    dfs = {
        'Gram': pd.read_csv('uvnet_solidmnist_font_subset_no_inorm_layer_probe_scores_with_err.csv'),
        'INorm Only': pd.read_csv('uvnet_solidmnist_font_subset_layer_probe_scores_with_err.csv'),
        'FNorm Only': pd.read_csv('uvnet_solidmnist_font_subset_face_norm_only_layer_probe_scores_with_err.csv'),
        'Mu & Sigma': pd.read_csv('uvnet_solidmnist_subset_feats_mu_sigma_layer_probe_scores_with_err.csv'),
        'Mu & Sigma + INorm': pd.read_csv('uvnet_solidmnist_subset_feats_mu_sigma_inorm_layer_probe_scores_with_err.csv'),
        'Mu': pd.read_csv('uvnet_solidmnist_subset_feats_mu_layer_probe_scores_with_err.csv'),
        'Mu + INorm': pd.read_csv('uvnet_solidmnist_subset_feats_mu_inorm_layer_probe_scores_with_err.csv'),
        'Mu & Sigma + INorm + FNorm': pd.read_csv('uvnet_solidmnist_subset_mu_sigma_inorm_concat_fnorm_layer_probe_scores_with_err.csv'),
    }

    for i, (version, df) in enumerate(dfs.items()):
        df = df.iloc[:1]
        xticks = np.arange(len(df)) * (len(dfs) + 1)
        plt.bar(x=i,
                height=df['linear_probe'],
                yerr=df['linear_probe_err'])

    ax = plt.gca()  # type: plt.Axes
    labels = dfs.keys()
    xticks = np.arange(len(labels))
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=90)
    plt.ylim([.8, 1.])
    plt.title('SolidMNIST Font Subset')
    plt.tight_layout()
    plt.savefig('inorm_plot_feats.pdf')
    plt.show()
