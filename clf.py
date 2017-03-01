import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def train_predict(x_tr, y_tr, x_te, y_te, clf):
    clf.fit(x_tr, y_tr)
    return clf.score(x_te, y_te)


def tp_pack(x_tr, y_tr, x_te, y_te, dim_reducer=None):
    dim = x_tr.shape[1]
    if dim_reducer:
        x_tr, x_te = dim_reducer.transform(x_tr), dim_reducer.transform(x_te)
        dim = dim_reducer.n_components

    # linear
    accuracy = linear_square(x_tr, y_tr, x_te, y_te)
    print('method = \'linear\', dim =', dim, ':', accuracy)

    # knn
    knn = KNeighborsClassifier()
    ks = [1, 5, 9]
    for k in ks:
        knn.set_params(n_neighbors=k)
        accuracy = train_predict(x_tr, y_tr, x_te, y_te, knn)
        print('method = \'knn\', k =', k, ', dim =', dim, ':', accuracy)


def linear_square(x_tr, y_tr, x_te, y_te):
    from sklearn.preprocessing import LabelBinarizer
    y_tr = LabelBinarizer().fit_transform(y_tr)
    from sklearn.linear_model import LinearRegression
    linear = LinearRegression().fit(x_tr, y_tr)
    y_pred = linear.predict(x_te)
    y_pred = y_pred.argmax(axis=1)
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_te, y_pred)

