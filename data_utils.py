import pickle
import numpy as np
import os


def load_cifar_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f,encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
        Y = np.array(Y)
        return X, Y


def load_cifar10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        print('loading train data from '+ f + '...')
        X, Y = load_cifar_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    f = os.path.join(ROOT, 'test_batch')
    print('loading test data from '+ f + '...\n')
    Xte, Yte = load_cifar_batch(f)
    return Xtr, Ytr, Xte, Yte


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


def split(x):
    x = x.transpose(3, 0, 1, 2)
    sp = x[0], x[1], x[2]
    sp = np.concatenate(sp)
    return sp
