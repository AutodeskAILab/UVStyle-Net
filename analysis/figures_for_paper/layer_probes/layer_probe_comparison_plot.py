import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':

    paths = {
        'UV-Net': 'uvnet_layer_probe_scores_with_err.csv',
        'Pointnet': 'psnet_layer_probe_scores_font_subset_with_err.csv',
        'Pointnet++': 'pointnet_layer_probe_scores_with_err.csv',
        'MeshCNN': 'meshcnn_layer_probe_scores_with_err.csv',
    }

    dfs = []
    for model, path in paths.items():
        df = pd.read_csv(path)
        df['model'] = model
        dfs.append(df)

    df = pd.concat(dfs).reset_index()

    fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes

    groups = df.groupby('model', sort=False)
    for i, (model, model_df) in enumerate(groups):
        ax.bar(x=model_df.index,
               height=model_df['linear_probe'],
               yerr=model_df['linear_probe_err'],
               label=model)

    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df['layer'], rotation='vertical', fontsize='small')
    ax.set_yticks(np.arange(7) / 5)
    ax.set_ylim(.7, 1)
    ax.legend(loc='lower right', fontsize='small')
    s = 0.9
    fig.set_size_inches(6 * s, 2.5 * s)
    fig.tight_layout()

    fig.savefig('layer_probe_comparison.pdf')
    fig.show()
