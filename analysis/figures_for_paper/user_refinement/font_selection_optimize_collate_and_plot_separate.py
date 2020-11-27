import os
from ast import literal_eval
from glob import glob
from pathlib import Path

import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figures_for_paper.single_letter_hits.single_letter_hits_plot import heatmap, annotate_heatmap
from util import Grams


def add_heatmap(arr, ax, pos, neg, sigs1=None, sigs2=None):
    im, cbar = heatmap(data=arr[list(reversed(np.arange(len(pos)))), :],
                       row_labels=list(reversed(pos)),
                       col_labels=neg,
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
                     texts=texts[list(reversed(np.arange(len(pos)))), :],
                     textcolors=['white', 'black'],
                     threshold=arr.mean())


def collate_df(root_dir):
    def load_df(file):
        df = pd.read_csv(file, sep=',').drop('Unnamed: 0', axis=1)
        _, trial, _, font, case = Path(file).stem.split('_')
        df['trial'] = int(trial)
        df['font'] = int(font)
        df['case'] = case
        return df

    files = glob(os.path.join(root_dir, '*.csv'))
    dfs = map(load_df, files)
    df = pd.concat(dfs)
    return df


def heatmap_and_std(ax, df, title, n_trials, pos, neg):
    # ax, err_ax = axes  # type: plt.Axes, plt.Axes
    arr = np.zeros([len(pos), len(neg)])
    err = np.zeros([len(pos), len(neg)])
    gain = np.zeros([len(pos), len(neg)])
    sigs1 = np.zeros_like(arr)
    sigs2 = np.zeros_like(arr)
    for p_i, p, n_i, n in [(p_i, p, n_i, n) for (p_i, p) in enumerate(pos) for (n_i, n) in enumerate(neg)]:
        try:
            arr[p_i, n_i] = df[df['pos_neg'] == str((p, n))]['score']
            err[p_i, n_i] = df[df['pos_neg'] == str((p, n))]['err']
            sigs1[p_i, n_i] = significance_test(mu_1=arr[0, 0],
                                                std_1=err[0, 0],
                                                mu_2=arr[p_i, n_i],
                                                std_2=err[p_i, n_i],
                                                n_trials=n_trials,
                                                q=0.9)
            sigs2[p_i, n_i] = significance_test(mu_1=arr[0, 0],
                                                std_1=err[0, 0],
                                                mu_2=arr[p_i, n_i],
                                                std_2=err[p_i, n_i],
                                                n_trials=n_trials,
                                                q=0.95)
            gain[p_i, n_i] = arr[p_i, n_i] / arr[0, 0] if arr[0, 0] > 0. else 1.
        except:
            continue
    ax.set_title(title)
    add_heatmap(arr, ax, pos, neg, sigs1=sigs1, sigs2=sigs2)
    ax.set_xlabel('No. of Negatives')
    ax.set_ylabel('No. of Positives')
    return arr, gain


def significance_test(mu_1, std_1, mu_2, std_2, n_trials, q=0.95):
    a = (std_1 ** 2) / n_trials
    b = (std_2 ** 2) / n_trials
    std_diff = np.sqrt(a + b)
    z = scipy.stats.norm.ppf(q)
    dist = z * std_diff
    return mu_2 - mu_1 > dist

def set_size(fig: plt.Figure):
    fig.set_size_inches(3, 4)

if __name__ == '__main__':
    df = collate_df('results_solidmnist_all_sub_mu_random_negatives')

    baseline_df = collate_df('results_solidmnist_all_sub_mu_fixed2')

    for p in [1, 2, 3, 4, 5, 10]:
        df = df.append(baseline_df[baseline_df['pos_neg'] == f'({p}, 0)'])

    df = df.groupby(['font', 'case', 'pos_neg']).mean().reset_index()

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
        fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
        arr, gain = heatmap_and_std(ax, font_df,
                                    title=f'{fonts[font]}_{case} (20 Trials)',
                                    n_trials=20,
                                    pos=pos,
                                    neg=neg)
        set_size(fig)
        fig.tight_layout()
        fig.savefig(f'{fonts[font]}_{case} (20 Trials) random negatives.png')
        fig.show()
        arrs.append(arr)
        gains.append(gain)

    fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
    num_fonts = len(df['font'].unique())
    print(f'{num_fonts} fonts')

    arr = np.mean(gains, axis=0)
    err = np.std(gains, axis=0)
    sigs1 = np.zeros_like(arr)
    sigs2 = np.zeros_like(arr)

    n_trials = 20 * num_fonts
    for p_i, p, n_i, n in [(p_i, p, n_i, n) for (p_i, p) in enumerate(pos) for (n_i, n) in enumerate(neg)]:
        sigs1[p_i, n_i] = significance_test(mu_1=arr[0, 0],
                                            std_1=err[0, 0],
                                            mu_2=arr[p_i, n_i],
                                            std_2=err[p_i, n_i],
                                            n_trials=n_trials,
                                            q=0.9)
        sigs2[p_i, n_i] = significance_test(mu_1=arr[0, 0],
                                            std_1=err[0, 0],
                                            mu_2=arr[p_i, n_i],
                                            std_2=err[p_i, n_i],
                                            n_trials=n_trials,
                                            q=0.95)
    add_heatmap(arr, ax, pos, neg, sigs1, sigs2)
    ax.set_title(f'Mean Gain ({num_fonts} Fonts)')
    ax.set_xlabel('No. of Negatives')
    ax.set_ylabel('No. of Positives')
    set_size(fig)
    fig.tight_layout()
    fig.savefig(f'Mean Gain ({num_fonts} Fonts) negative randoms.png')
    fig.show()
