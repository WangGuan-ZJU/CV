import cv2
import numpy as np


def get_dsift(imgs, kps):

    sift_detector = cv2.xfeatures2d.SIFT_create()
    n = len(kps)
    print(n)
    dsift = [sift_detector.compute(img.astype('uint8'), kps)[1] for img in imgs]
    dsift = np.array(dsift).reshape(imgs.shape[0] * n, 128)
    return dsift


def build_dicts(dsift, n_clusters):
    from sklearn.cluster import MiniBatchKMeans
    kmean = MiniBatchKMeans(n_clusters=n_clusters, random_state=0).fit(dsift.astype('float64'))
    dicts = kmean.cluster_centers_.astype('float')
    return dicts


def sparse_encode(data, dicts):
    from sklearn.decomposition import sparse_encode
    code = sparse_encode(data, dicts, algorithm='lasso_cd', alpha=0.35)
    return code


def average_pooling(data, shape):
    data = data.reshape(shape).transpose(0, 2, 1)
    data = np.mean(data, axis=2)
    print('avg pooling shape:', data.shape)
    return data
