import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    uvnet = pd.read_csv('uvnet_layer_probe_scores_with_err.csv')
    pointnet = pd.read_csv('pointnet_layer_probe_scores_with_err.csv')
    meshcnn = pd.read_csv('meshcnn_layer_probe_scores_with_err.csv')

    uvnet['model'] = 'UV-Net'
    pointnet['model'] = 'Pointnet++'
    meshcnn['model'] = 'MeshCNN'

    df = pd.concat([uvnet, pointnet, meshcnn])

    fig, axes = plt.subplots(nrows=3)  # type: plt.Figure, np.ndarray

    groups = df.groupby('model', sort=False)

    for i, (model, model_df) in enumerate(groups):
        ax = axes[i]
        ax.set_title(model)
        ax.bar(np.arange(len(model_df)), model_df['linear_probe'],
               yerr=model_df['linear_probe_err'])
        # ax.errorbar(np.arange(len(model_df)), model_df['linear_probe'],
        #             yerr=model_df['linear_probe_err'])
        ax.set_xticks(np.arange(len(model_df)))
        ax.set_yticks(np.arange(7) / 5)
        ax.set_ylim([.6, 1])
        # ax.plot([-.5, len(model_df) - .5], [.25, .25], '--', color='black')
        ax.set_xticklabels(model_df['layer'], rotation='vertical')
    fig.set_size_inches(6, 6)
    fig.tight_layout()
    fig.savefig('layer_probe_comparison.pdf')
    fig.show()
