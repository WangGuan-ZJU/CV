import numpy as np
import mxnet as mx
import logging
import pickle
import os


def load_cifar_batch(filename):
    logging.info('loading data from %s ...'% filename)
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
        img = data['data']
        lbl = data['labels']
        # reshape and normalize
        img = np.reshape(img, (10000, 3, 32, 32)).astype('float32') / 255
        lbl = np.array(lbl)
        return img, lbl


def load_cifar10(path):
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(path, 'data_batch_%d' % (b,))
        X, Y = load_cifar_batch(f)
        xs.append(X)
        ys.append(Y)
    train_img = np.concatenate(xs)
    train_lbl = np.concatenate(ys)
    f = os.path.join(path, 'test_batch')
    var_img, val_lbl = load_cifar_batch(f)
    return train_img, train_lbl, var_img, val_lbl


def fast_alexnext(num_classes):
    data = mx.symbol.Variable('data')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, name='conv1', pad=(2, 2), kernel=(5, 5), num_filter=32)
    bn1 = mx.symbol.BatchNorm(data=conv1)
    relu1 = mx.symbol.Activation(data=bn1, act_type='relu')
    pool1 = mx.symbol.Pooling(data=relu1, pool_type='max',
                              kernel=(3, 3), stride=(2, 2))
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, name='conv2', pad=(2, 2), kernel=(5, 5), num_filter=32)
    bn2 = mx.symbol.BatchNorm(data=conv2)
    relu2 = mx.symbol.Activation(data=bn2, act_type='relu')
    pool2 = mx.symbol.Pooling(data=relu2, pool_type='avg',
                              kernel=(3, 3), stride=(2, 2))
    # third conv
    conv3 = mx.symbol.Convolution(data=pool2, name='conv3', pad=(2, 2), kernel=(5, 5), num_filter=64)
    bn3 = mx.symbol.BatchNorm(data=conv3)
    relu3 = mx.symbol.Activation(data=bn3, act_type='relu')
    pool3 = mx.symbol.Pooling(data=relu3, pool_type='avg',
                              kernel=(3, 3), stride=(2, 2), name='final_pool')
    # first fullc
    flatten = mx.symbol.Flatten(data=pool3)
    fc1 = mx.symbol.FullyConnected(data=flatten, name='full', num_hidden=64)
    relu4 = mx.symbol.Activation(data=fc1, act_type="relu")
    # second fullc
    fc2 = mx.symbol.FullyConnected(data=relu4, num_hidden=num_classes)
    # softmax
    softmax = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return softmax


def main():
    train_img, train_lbl, val_img, val_lbl = load_cifar10('data')
    logging.info('data loaded.')
    # create data iter
    batch_size = 100
    train_iter = mx.io.NDArrayIter(train_img, train_lbl, batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(val_img, val_lbl, batch_size)
    logging.info('data iterator created.')
    # cnn for cifar-10
    net = fast_alexnext(10)
    # visualize the network
    # shape = {'data': (50000, 3, 32, 32)}
    # mx.viz.plot_network(net, shape=shape).view()
    # Output may vary
    model = mx.model.FeedForward(
        ctx = mx.gpu(0),
        symbol=net,
        num_epoch=100,
        learning_rate=0.1)
    model.fit(
        X=train_iter,
        eval_data=val_iter,
        batch_end_callback=mx.callback.Speedometer(batch_size, 200)
    )
    model.save('model')
    score = model.score(val_iter)
    print(score)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    main()

