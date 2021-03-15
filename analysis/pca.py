import numpy as np
from sklearn.decomposition import PCA

from util import Grams, get_pca_3_70

if __name__ == '__main__':
    data_root = 'abc_sub_mu_only'
    grams = Grams(data_root=data_root)

    get_pca_3_70()
    PCA(70)