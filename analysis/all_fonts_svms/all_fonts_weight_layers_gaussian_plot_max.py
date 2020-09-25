import pathlib
from ast import literal_eval
from glob import glob
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd


def load_df(file):
    df = pd.read_csv(file, sep=',')
    file = pathlib.Path(file)
    df['font'] = file.stem.split('_')[-1]
    return df


if __name__ == '__main__':
    df = pd.concat([load_df(file) for file in glob('gaussian_weights_scores/*.csv')],
                   axis=0)
    df['mu'], df['sigma'] = zip(*df['mu_sigma'].apply(literal_eval))

    df = df.groupby(['font', 'mu_sigma'], as_index=False).mean()
    df = df.groupby(['font']).max()  # type: pd.DataFrame

    df.sort_values('score', ascending=False, inplace=True)

    fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
    ax.set_title('Grid Search Mu and Sigma')
    ax.set_ylim(top=0.9)
    ax.set_xlabel('Fonts (sorted by score)')
    ax.set_ylabel('Mean Hits@10 Score')
    ax.plot([0, 167], [0.06, 0.06], '--', color='black')
    ax.bar(np.arange(len(df)), df['score'])
    fig.savefig('mean_hits_at_10_histogram.pdf')
    fig.show()
