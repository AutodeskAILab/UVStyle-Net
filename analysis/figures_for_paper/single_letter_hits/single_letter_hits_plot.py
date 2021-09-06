import matplotlib as mtp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




if __name__ == '__main__':
    titles = {
        'uvnet': 'UV-Net',
        'pointnet': 'Pointnet++'
    }
    styles = {
        'slanted': 'Slanted',
        'serif': 'Serif',
        'nocurve': 'No Curves'
    }

    fig, axes = plt.subplots(nrows=3, ncols=2)

    for col, model in enumerate(['uvnet', 'pointnet']):
        for row, style in enumerate(['slanted', 'serif', 'nocurve']):
            df = pd.read_csv(f'{model}_{style}_svm.txt', sep=',')
            values = df[['1', '2', '3', '4', '5']].to_numpy()
            ax = axes[row, col]

            im, cbar = heatmap(values,
                               ax=ax,
                               row_labels=list(map(str, np.arange(6))),
                               col_labels=list(map(str, np.arange(1, 6))),
                               cmap="viridis",
                               vmin=0,
                               vmax=1)
            cbar.remove()
            texts = annotate_heatmap(im=im, valfmt="{x:.2f}",
                                     textcolors=['white', 'black'],
                                     threshold=.5)
            if row == 0:
                ax.annotate('No. of Positives', xy=(0.5, 1.1), xytext=(0, 5),
                            xycoords='axes fraction', textcoords='offset points',
                            size='medium', ha='center', va='baseline')
                ax.annotate(titles[model], xy=(0.5, 1.2), xytext=(0, 5),
                            xycoords='axes fraction', textcoords='offset points',
                            size='large', ha='center', va='baseline')
            if col == 0:
                ax.set_ylabel('No. of Negatives')
                ax.annotate(styles[style], xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                            xycoords=ax.yaxis.label, textcoords='offset points',
                            size='large', ha='right', va='center',
                            rotation=90)

    fig.set_size_inches(5, 8)
    fig.subplots_adjust(wspace=0.2)
    fig.tight_layout()
    fig.savefig('single_letter_hits_svm.pdf')
    fig.show()
