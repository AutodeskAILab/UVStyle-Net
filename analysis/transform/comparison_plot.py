import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('results.csv', sep=',', header=None,
                     names=['model', 'trial', 'test_loss', 'test_acc', 'checkpoint'])
    mean = df.groupby(['model']).mean()
    std = df.groupby(['model']).std()

    fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
    xticks = np.arange(len(mean))
    ax.bar(xticks, mean['test_acc'], yerr=std['test_acc'])
    ax.plot([-0.5, len(mean) - .5], [0.006, 0.006], '--', color='black')
    ax.set_xticks(xticks)
    ax.set_xticklabels(mean.index, rotation=90)
    fig.set_size_inches(8, 6)
    fig.tight_layout()
    fig.show()
