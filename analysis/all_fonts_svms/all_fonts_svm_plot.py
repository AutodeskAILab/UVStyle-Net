import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_bar_plot(df, filename):
    fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes

    x_pos = np.arange(len(df))
    ax.bar(x=x_pos,
           height=df['acc'],
           yerr=df['std'])

    ax.set_xlabel('Font')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df.index, size='xx-small', rotation=90)

    ax.set_ylabel('Balanced Accuracy')
    ax.set_ylim(.49, .51)
    ax.set_xlim(-1, x_pos.max() + 1)

    ax.set_title('Balanced Accuracy Scores (SVM) Using Cosine Distance\n'
                 f'Mean: {df["acc"].mean():.2f} +\- {df["std"].mean():.2f}')
    fig.set_size_inches(40, 3)
    fig.tight_layout()
    fig.savefig(filename)
    fig.show()


if __name__ == '__main__':
    df = pd.read_csv('all_fonts_svm_results.csv', sep=',')
    save_bar_plot(df, 'all_fonts_svm_plot.pdf')
