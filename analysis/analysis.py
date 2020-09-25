import umap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    run_id = 'Jun25_12-33-08_C02ZX3Q7MD6R/00000'
    for layer in range(1):
        print(f'Layer {layer + 1}: loading data...')
        meta = pd.read_csv(f'runs/{run_id}/layer{layer + 1}/metadata.tsv',
                           sep='\t')
        layer_raw = pd.read_csv(f'runs/{run_id}/layer{layer + 1}/tensors.tsv',
                                sep='\t',
                                header=None)
        layer_raw.to_pickle(f'runs/{run_id}/layer{layer + 1}/tensors.pd')
        # layer_raw = pd.read_pickle(f'runs/{run_id}/layer{layer + 1}/tensors.pd')

        # print(f'Layer {layer}: scaling...')
        # scaler = StandardScaler()
        # layer_scaled = scaler.fit_transform(layer_raw)

        print(f'Layer {layer + 1}: fitting UMAP...')
        reducer = umap.UMAP(verbose=True, n_epochs=400)
        layer_emb = reducer.fit_transform(layer_raw)

        cmap = plt.get_cmap('Set1')

        print(f'Layer {layer + 1}: plotting...')
        for feature in meta.columns:
            plt.title(f'Layer {layer + 1}: {feature}')
            plt.scatter(layer_emb[:, 0],
                        layer_emb[:, 1],
                        s=1,
                        alpha=0.5,
                        c=cmap(meta[feature]))
            plt.savefig(f'dump/plots/layer_{layer + 1}_{feature}.png')
            # plt.show()
