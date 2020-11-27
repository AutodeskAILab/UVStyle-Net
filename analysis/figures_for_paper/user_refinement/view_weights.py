from ast import literal_eval
from glob import glob
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from util import Grams


def load_df(file):
    filename = Path(file).stem
    _, trial, _, font, case = filename.split('_')
    df = pd.read_csv(file, sep=',')
    df['trial'] = int(trial)
    df['font'] = int(font)
    df['case'] = case
    df['weights'] = df['weights'].apply(literal_eval)
    return df


if __name__ == '__main__':
    results_path = 'results_solidmnist_all_sub_mu_random_negatives'
    files = glob(f'{results_path}/*.log')
    df = pd.concat(map(load_df, files))

    baseline_path = 'results_solidmnist_all_sub_mu_fixed2'
    baseline_files = glob(f'{baseline_path}/*.log')
    baseline_df = pd.concat(map(load_df, baseline_files))

    for p in [1, 2, 3, 4, 5, 10]:
        df = df.append(baseline_df[(baseline_df['positives'] == p)
                                   & (baseline_df['negatives'] == 0)])

    df = df.sort_values(['negatives', 'negatives'], ascending=[True, False])

    grams = Grams('../../uvnet_data/solidmnist_all_sub_mu')
    fonts = {
        i: font_name for i, font_name in zip(grams.labels, map(lambda f: f[2:-10], grams.graph_files))
    }

    groups = df.groupby(['font', 'case'])
    for (font, case), group_df in groups:
        fig, axes = plt.subplots(nrows=group_df['positives'].nunique(),
                                 ncols=group_df['negatives'].nunique(),
                                 sharex='all',
                                 sharey='all')  # type: plt.Figure, np.ndarray

        for row, p in enumerate(group_df['positives'].unique()):
            for col, n in enumerate(group_df['negatives'].unique()):
                ax = axes[group_df['positives'].nunique() - 1 - row, col]  # type: plt.Axes

                weights = group_df[(group_df['positives'] == p) & (group_df['negatives'] == n)]['weights']
                weights = np.array(weights.tolist())
                mean = weights.mean(axis=0)
                std = weights.std(axis=0)
                xticks = np.arange(len(mean))
                ax.errorbar(x=xticks,
                            y=mean,
                            yerr=std)
                if row == 0:
                    ax.set_xlabel(n)
                if col == 0:
                    ax.set_ylabel(p)

        fig.suptitle(f'{fonts[font]}_{case}')
        fig.tight_layout()
        fig.savefig(f'{fonts[font]}_{case}_weights_random_negatives.png')
        fig.show()
