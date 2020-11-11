import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('dimension_reduction_probe_results.csv')
    #idf_content = pd.read_csv('dimension_reduction_probe_results_content.csv')
    #df = pd.concat([df, df_content], axis=0, ignore_index=True)

    df['dims'] = df['dims'].apply(lambda d: str(int(d)) if d > 0 else 'Original')

    layer_names = [
        '0_feats (36)',
        '1_conv (4,096)',
        '2_conv (16,384)',
        '3_conv (65,536)',
        '4_fc (4,096)',
        '5_GIN (4,096)',
        '6_GIN (4,096)',
        'content (256)'
    ]

    fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
    xticks = np.arange(df['dims'].nunique())
    for i, (layer, group_df) in enumerate(df.groupby(['layer'])):
        ax.plot(xticks,
                group_df['score'],
                label=layer_names[layer])
        ax.fill_between(xticks,
                        group_df['score'] - group_df['err'],
                        group_df['score'] + group_df['err'],
                        alpha=0.1)

    ax.legend()
    ax.set_xlabel('Reduced Dimensions')
    ax.set_xticks(xticks)
    ax.set_xticklabels(['None', '70', '25', '10', '3'])
    ax.set_ylabel('Linear Probe Score')
    fig.tight_layout()
    fig.savefig('dimension_reduction_plot.pdf')
    fig.show()
