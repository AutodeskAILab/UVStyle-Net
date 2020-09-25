import numpy as np
import matplotlib as mtp
import matplotlib.pyplot as plt
import pandas as pd


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
