import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('uvnet_labels.csv', index_col='label')
    df['font'] = df['file_names'].apply(lambda f: f.split(' '))
    last = ['_']
    last2 = ['_']
    matches = []
    for i in range(len(df)):
        font = df.iloc[i]['font']
        if len(font) > 1:
            base = font[:-1]
            if base == last or base == last2:
                matches.append(base)
                last = base
                last2 = font
                continue
        matches.append('')
        last = font

    df['match'] = matches
    df = df[df['match'] == '']

    df['file_names'].to_csv('filtered_font_labels.csv', index=None, header=False)