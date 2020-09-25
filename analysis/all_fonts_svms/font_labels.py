import pandas as pd

from util import Grams

if __name__ == '__main__':
    grams = Grams('../uvnet_data/solidmnist_all')
    df = pd.DataFrame({
        'file_names': grams.graph_files,
        'label': grams.labels
    }).sort_values('label')  # type: pd.DataFrame
    df = df.groupby('label').first()
    df['file_names'] = df['file_names'].apply(lambda f: f[2:-10])
    df.to_csv('uvnet_labels.csv', sep=',')
