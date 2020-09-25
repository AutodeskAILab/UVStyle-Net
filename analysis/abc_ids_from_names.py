import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('uvnet_data/abc_all/graph_files.txt', names=['name'], header=None)
    while True:
        name = input()
        try:
            print(df[df['name'] == name + '.bin'].index.tolist()[0])
        except Exception as e:
            print('Cannot find ', name)

