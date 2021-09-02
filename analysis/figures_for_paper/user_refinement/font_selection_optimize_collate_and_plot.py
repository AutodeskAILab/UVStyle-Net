import os
from ast import literal_eval
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from figures_for_paper.single_letter_hits.single_letter_hits_plot import heatmap, annotate_heatmap
from util import Grams


def add_heatmap(arr, ax, pos, neg, sigs1=None, sigs2=None):
    im, cbar = heatmap(data=arr[list(reversed(np.arange(len(neg)))), :],
                       row_labels=list(reversed(neg)),
                       col_labels=pos,
                       ax=ax,
                       cmap="viridis",
                       label_top=False)
    if sigs1 is None:
        sigs1 = np.zeros_like(arr)
    if sigs2 is None:
        sigs2 = np.zeros_like(arr)

    def create_text(arr_sig):
        mean, sig1, sig2 = arr_sig
        return f'{mean:.2f}{"*" * int(sig1 + sig2)}'

    texts = np.array(list(map(create_text, zip(arr.flatten(), sigs1.flatten(), sigs2.flatten()))))
    texts = texts.reshape(arr.shape)

    annotate_heatmap(im=im, valfmt="{x}",
                     texts=texts[list(reversed(np.arange(len(neg)))), :],
                     textcolors=['white', 'black'],
                     threshold=arr.mean())


def collate_df(root_dir):
    def load_df(file):
        df = pd.read_csv(file, sep=',').drop('Unnamed: 0', axis=1)
        f = Path(file).stem.split('_')
        df[f[0]] = int(f[1])
        df[f[2]] = int(f[3])
        df['case'] = f[4]
        return df

    files = glob(os.path.join(root_dir, '*.csv'))
    dfs = map(load_df, files)
    df = pd.concat(dfs)
    return df


def heatmap_and_std(ax, df, title, n_trials, pos, neg):
    # ax, err_ax = axes  # type: plt.Axes, plt.Axes
    arr = np.zeros([len(neg), len(pos)])
    err = np.zeros([len(neg), len(pos)])
    gain = np.zeros([len(neg), len(pos)])
    sigs1 = np.zeros_like(arr)
    sigs2 = np.zeros_like(arr)
    for p_i, p, n_i, n in [(p_i, p, n_i, n) for (p_i, p) in enumerate(pos) for (n_i, n) in enumerate(neg)]:
        try:
            arr[n_i, p_i] = df[df['pos_neg'] == str((p, n))]['score']
            err[n_i, p_i] = df[df['pos_neg'] == str((p, n))]['err']
            sigs1[n_i, p_i] = significance_test(mu_1=arr[0, 0],
                                                std_1=err[0, 0],
                                                mu_2=arr[n_i, p_i],
                                                std_2=err[n_i, p_i],
                                                n_trials=n_trials,
                                                q=0.9)
            sigs2[n_i, p_i] = significance_test(mu_1=arr[0, 0],
                                                std_1=err[0, 0],
                                                mu_2=arr[n_i, p_i],
                                                std_2=err[n_i, p_i],
                                                n_trials=n_trials,
                                                q=0.95)
            gain[n_i, p_i] = arr[n_i, p_i] / arr[0, 0] if arr[0, 0] > 0. else 1.
        except:
            continue
    ax.set_title(title)
    add_heatmap(arr, ax, pos, neg, sigs1=sigs1, sigs2=sigs2)
    ax.set_xlabel('No. of Positives')
    ax.set_ylabel('No. of Negatives')
    # err_ax.set_title(title + ' Increase')
    # add_heatmap(gain, err_ax)
    return arr, gain


def significance_test(mu_1, std_1, mu_2, std_2, n_trials, q=0.95):
    a = (std_1 ** 2) / n_trials
    b = (std_2 ** 2) / n_trials
    std_diff = np.sqrt(a + b)
    z = scipy.stats.norm.ppf(q)
    dist = z * std_diff
    return mu_2 - mu_1 > dist


if __name__ == '__main__':
    df = collate_df('results_solidmnist_all_sub_mu_only')
    df = df.groupby(['font', 'case', 'pos_neg']).mean().reset_index()

    fig, axes = plt.subplots(nrows=len(df['font'].unique()) * 2,
                             ncols=1,
                             squeeze=True)  # type: plt.Figure, np.ndarray
    grams = Grams('../../uvnet_data/solidmnist_all_fnorm')
    fonts = {
        i: font_name for i, font_name in zip(grams.labels, map(lambda f: f[2:-10], grams.graph_files))
    }
    pos, neg = zip(*df['pos_neg'].apply(literal_eval).values)
    pos = sorted(list(set(pos)))
    neg = sorted(list(set(neg)))
    arrs = []
    gains = []
    for i, ((font, case), font_df) in enumerate(df.groupby(['font', 'case'])):
        arr, gain = heatmap_and_std(axes[i], font_df,
                                    title=f'{font}: {fonts[font]}_{case} (20 Trials)',
                                    n_trials=20,
                                    pos=pos,
                                    neg=neg)
        arrs.append(arr)
        gains.append(gain)
    s = .9
    fig.set_size_inches(7 * s, 60 * s)
    fig.tight_layout()
    fig.savefig('svm_user_optimization_sub_mu_only_upper_lower.pdf')
    # fig.show()

    fig, axes = plt.subplots(ncols=2, squeeze=True)  # type: plt.Figure, np.ndarray
    all_df = df.groupby('pos_neg').mean().reset_index()
    num_fonts = len(df['font'].unique())
    # heatmap_and_std(axes, all_df, f'Mean ({num_fonts} Fonts)', n_trials=num_fonts)
    add_heatmap(arr=np.mean(np.stack(gains), axis=0),
                ax=axes[0], pos=pos, neg=neg)
    add_heatmap(arr=np.std(np.stack(gains), axis=0),
                ax=axes[1], pos=pos, neg=neg)
    s = .8
    fig.set_size_inches(10 * s, 6 * s)
    fig.tight_layout()
    fig.savefig('svm_user_optimization_sub_mu_only_all_upper_lower.pdf')
    fig.show()
    # fig.show()
