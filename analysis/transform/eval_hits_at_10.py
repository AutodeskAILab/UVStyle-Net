import pandas as pd
import numpy as np
from progressbar import progressbar
import matplotlib.pyplot as plt

from all_fonts_svms.all_fonts_weight_layers_gaussian import hits_at_k_score

if __name__ == '__main__':
    embeddings = np.load('embeddings_1026.npy')

    graph_files = np.loadtxt('graph_files_1026.txt', delimiter=',', dtype=np.str)
    labels = pd.DataFrame(graph_files)[0].apply(lambda f: f[2:-10]).astype('category').cat.codes.values

    fonts = []
    scores = []
    errs = []
    for font in progressbar(set(labels)):
        score, err = hits_at_k_score(embeddings, labels, target=font)
        fonts.append(font)
        scores.append(score)
        errs.append(err)

    df = pd.DataFrame({
        'font': fonts,
        'score': scores,
        'err': errs
    })

    df.sort_values('score', ascending=False, inplace=True)

    fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
    ax.set_ylim(top=1.)
    ax.set_title('Contrastive Crop Only (20)')
    ax.set_xlabel('Fonts (sorted by score)')
    ax.set_ylabel('Mean Hits@10 Score')
    ax.plot([-.5, len(df) - .5], [0.06, 0.06], '--', color='black')
    ax.bar(np.arange(len(df)), df['score'])
    fig.show()

    font_names = np.loadtxt('../all_fonts_svms/filtered_font_labels.csv', delimiter=',', dtype=np.str)
    for i in range(5):
        row = df.iloc[int((i / 4) * (len(df) - 1))]
        font = int(row['font'])
        name = font_names[font]
        score = row['score']
        print(f'q{i}: {font} ({name}) - score: {score}')
