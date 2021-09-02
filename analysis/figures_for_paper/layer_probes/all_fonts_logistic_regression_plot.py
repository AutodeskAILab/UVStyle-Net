import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    n_classes = 167
    df = pd.read_csv('all_fonts_logistic_regression_results.csv', sep=',')
    layer_scores = df.groupby('layer').mean().reset_index()
    layer_errs = df.groupby('layer').std().reset_index()

    fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes

    ax.set_title('SolidMNIST All Fonts')
    ax.bar(x=layer_scores['layer'],
           height=layer_scores['score'],
           yerr=layer_errs['score'])
    ax.plot([-.5, len(layer_scores) - .5],
            [1 / n_classes, 1/n_classes],
            '--', color='black')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Classification Accuracy')

    fig.tight_layout()
    fig.savefig('all_fonts_logistic_regression.pdf')
    fig.show()
