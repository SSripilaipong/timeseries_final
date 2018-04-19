import os
import numpy as np

def load_data(name, data_dir, group='TRAIN'):
    with open(os.path.join(data_dir, '{}_{}'.format(name, group))) as file:
        X, y = list(), list()
        for line in file:
            d = list(map(float, line.split()))
            y_, x_ = d[0], d[1:]

            X.append(x_)
            y.append(y_)
    return np.array(X), np.array(y)