import shutil
from glob import glob

if __name__ == '__main__':
    # df = pd.read_csv('uvnet_data/abc/graph_names.txt', header=None, names=['name'])
    #
    # for name in df['name']:
    #     try:
    #         shutil.move(f'/Users/t_meltp/abc/pngs/{name.strip()}.png', '/Users/t_meltp/abc/pngs/test/')
    #     except Exception as e:
    #         print(e, file=sys.stderr)

    files = glob('/Users/t_meltp/abc/pngs/*.png', recursive=False)
    for file in files:
        shutil.move(file, '/Users/t_meltp/abc/pngs/train/')
