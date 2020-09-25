import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    without_inorm = pd.read_csv('uvnet_no_inorm_layer_probe_scores_with_err.csv')
    with_inorm = pd.read_csv('../layer_probes/uvnet_layer_probe_scores_with_err.csv')

    xticks = np.arange(len(with_inorm))

    plt.bar(x=xticks - 0.2,
            height=without_inorm['linear_probe'],
            width=0.4,
            yerr=without_inorm['linear_probe_err'],
            label='Without INorm')
    plt.bar(x=xticks + 0.2,
            height=with_inorm['linear_probe'],
            width=0.4,
            yerr=with_inorm['linear_probe_err'],
            label='With INorm')
    plt.legend()
    ax = plt.gca()  # type: plt.Axes
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels=[
        '0_feats',
        '1_conv',
        '2_conv',
        '3_conv',
        '4_fc',
        '5_GIN',
        '6_GIN'
    ])
    plt.plot([5.2, 5.2], [0.9629, 0.963], '-', color='black')
    plt.ylim([.8, 1.])
    plt.tight_layout()
    plt.savefig('inorm_plot.pdf')
    plt.show()
