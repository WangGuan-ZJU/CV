from data_utils import *
from feature_utils import *
from clf import *


if __name__ == '__main__':
    x_tr, y_tr, x_te, y_te = load_cifar10('data')
    x_tr_gray, x_te_gray = rgb2gray(x_tr), rgb2gray(x_te)

    step, size = 2, 3
    lboundx, rboundx, lboundy, rboundy = 0, 32, 0, 32
    kps = [cv2.KeyPoint(x, y, size)
           for x in range(size + lboundx, rboundx - size, step)
           for y in range(size + lboundy, rboundy - size, step)]
    dsift_tr, dsift_te = get_dsift(x_tr_gray, kps), get_dsift(x_te_gray, kps)
    print('dense sift extracted.')

    from sklearn.preprocessing import Normalizer
    norm = Normalizer().fit(dsift_tr)
    dsift_tr, dsift_te = norm.transform(dsift_tr), norm.transform(dsift_te)
    print('normalization done.')

    n_features = 180
    dicts = build_dicts(dsift_tr, n_features)
    print('dictionary built.')

    code_tr, code_te = sparse_encode(dsift_tr, dicts), sparse_encode(dsift_te, dicts)
    print('sparse encoding done.')

    s1, s2 = (x_tr.shape[0], len(kps), n_features), (x_te.shape[0], len(kps), n_features)
    code_tr, code_te = average_pooling(code_tr, s1), average_pooling(code_te, s2)
    print('average pooling done.')

    tp_pack(code_tr, y_tr, code_te, y_te, None)
