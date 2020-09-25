import numpy as np
import matplotlib.axes
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
from typing import List


def error_curve(ax: matplotlib.axes.Axes, ys, ys_stds, xs=None, style: str = 'band', **kwargs):
    ys = np.asarray(ys)
    ys_stds = np.asarray(ys_stds)
    label = kwargs.get('label')
    alpha = kwargs.get('alpha', 0.2)
    color = kwargs.get('c')
    bandcolor = kwargs.get('bandcolor', color)
    barcolor = kwargs.get('barcolor', color)
    if xs is None:
        xs = np.arange(ys.size)
    ax.plot(xs, ys, label=label, c=color)
    if style == 'band':
        ax.fill_between(x=xs, y1=(ys - ys_stds), y2=(ys + ys_stds), alpha=alpha, facecolor=bandcolor)
    elif style == 'bar':
        ax.errorbar(x=xs, y=ys, yerr=ys_stds, alpha=alpha, ecolor=barcolor)
    else:
        raise ValueError('Unknown error style {}, expected one of ("band", "bar")'.format(style))


def confusion_matrix(ax: matplotlib.axes.Axes, true: np.ndarray, pred: np.ndarray,
                     classes: List[str] = None, normalize: bool = True,
                     cmap: matplotlib.colors.Colormap = plt.cm.Blues, title: str = ''):
    conf = metrics.confusion_matrix(true, pred)
    if normalize:
        conf = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]
    class_labels = unique_labels(true, pred)
    if classes is not None:
        class_labels = [classes[i] for i in class_labels]

    ticks = np.arange(len(class_labels))
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.invert_yaxis()

    im = ax.matshow(conf, interpolation='nearest', cmap=cmap)
    ax.set(xticks=ticks, xticklabels=class_labels, yticks=ticks, yticklabels=class_labels)
    #ax.figure.colorbar(im, ax=ax)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = conf.max() / 2.
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            ax.text(j, i, format(conf[i, j], fmt),
                    ha="center", va="center",
                    color="white" if conf[i, j] > thresh else "black")