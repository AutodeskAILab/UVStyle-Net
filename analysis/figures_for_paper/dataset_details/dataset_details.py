from solid_mnist import SolidMNIST, SolidMNISTSubset, SolidMNISTSingleLetter
import pandas as pd


def count_fonts(dset: SolidMNIST):
    return len(set(map(lambda f: f.name[2:-10], dset.graph_files)))


def count_classes(dset: SolidMNIST):
    return len(set(map(lambda f: f.name[0], dset.graph_files)))


if __name__ == '__main__':
    data_root = '/home/ubuntu/NURBSNet/dataset/bin'
    dsets = {
        'Complete': SolidMNIST,
        'Font Subset (4)': SolidMNISTSubset,
        'Single Letter (G)': SolidMNISTSingleLetter
    }

    columns = ['Dataset', 'Split', 'Examples', 'Classes', 'Fonts']
    df = pd.DataFrame(columns=columns)
    for dset_name, dset in dsets.items():
        for split in ['train', 'val', 'test']:
            d = dset(root_dir=data_root, split=split)
            df = df.append(pd.DataFrame(
                columns=columns,
                data=[(dset_name, split, len(d), count_classes(d), count_fonts(d))])
            )

    result = df.groupby('Dataset').agg(lambda row: r' / '.join(map(str, row)))
    print(result.drop('Split', axis=1).to_latex())
