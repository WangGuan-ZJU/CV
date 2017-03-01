from clf import *
from data_utils import *
import numpy as np


def tp_pack_pack(Xtr, Ytr, Xte, Yte, normalize=False):
    # reshape
    s1, s2 = Xtr.shape, Xte.shape
    p1, p2 = np.product(s1[1:]), np.product(s2[1:])
    Xtr, Xte = Xtr.reshape(s1[0], p1), Xte.reshape(s2[0], p2)
    print('reshape success.')

    # normalize
    if normalize:
        from sklearn.preprocessing import Normalizer
        norm = Normalizer().fit(Xtr)
        Xtr, Xte = norm.transform(Xtr), norm.transform(Xte)
        print('normalize success.')

    dims = [200, 300, 500]
    for dim in dims:
        # random projection
        print('RP:')
        from sklearn.random_projection import SparseRandomProjection
        dim_reducer = SparseRandomProjection(dim).fit(Xtr)
        tp_pack(Xtr, Ytr, Xte, Yte, dim_reducer)
    for dim in dims:
        # svd
        print('SVD:')
        from sklearn.decomposition import TruncatedSVD
        dim_reducer = TruncatedSVD(dim).fit(Xtr)
        tp_pack(Xtr, Ytr, Xte, Yte, dim_reducer)


if __name__ == '__main__':
    x_tr, y_tr, x_te, y_te = load_cifar10('data')
    x_tr_gray, x_te_gray = rgb2gray(x_tr), rgb2gray(x_te)

    print('train and predict color data(normalize=True):')
    tp_pack_pack(x_tr, y_tr, x_te, y_te, True)
    print('train and predict gray data(normalize=True):')
    tp_pack_pack(x_tr_gray, y_tr, x_te_gray, y_te, True)

    print('train and predict color data(normalize=False):')
    tp_pack_pack(x_tr, y_tr, x_te, y_te)
    print('train and predict gray data(normalize=False):')
    tp_pack_pack(x_tr_gray, y_tr, x_te_gray, y_te)

