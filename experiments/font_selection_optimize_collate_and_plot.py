import os
import sys
from argparse import ArgumentParser
from ast import literal_eval
from glob import glob
from pathlib import Path

import matplotlib as mtp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(file_dir)
sys.path.append(project_root)
from utils import Grams


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", label_top=True, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()  # type: plt.Axes

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    if label_top:
        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=None, ha="right",
    #          rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, texts=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mtp.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    if texts is None:
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)
    else:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                im.axes.text(j, i, valfmt(texts[i, j], None), **kw)
    return texts


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
    parser = ArgumentParser()
    parser.add_argument('--exp_name', default='SolidLETTERS-all', type=str,
                        help='experiment name - font scores will be read from'
                             'this directory (default: SolidLETTERS-all)')
    args = parser.parse_args()

    df = collate_df(args.exp_name)

    df = df.groupby(['font', 'case', 'pos_neg']).mean().reset_index()

    grams = Grams(os.path.join(project_root, 'data', 'SolidLETTERS', 'uvnet_grams', 'all'))
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
        fig.savefig(f'{fonts[font]}_{case} (20 Trials) random negatives.pdf')
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
    fig.savefig(f'Mean Gain ({num_fonts} Fonts) negative randoms.pdf')
    fig.show()
