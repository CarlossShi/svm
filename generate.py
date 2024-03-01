import numpy as np
import os
import pickle
import argparse


def generate(l, w, b, shrinkage=0.5, dtype='float64'):
    assert dtype in {'float16', 'float32', 'float64'}
    w = np.array(w)
    dtype = getattr(np, dtype)
    X = (np.random.uniform(low=-1, high=1, size=(l, len(w))) - b * w / np.inner(w, w)).astype(dtype)  # [NumPy calculate square of norm 2 of vector](https://stackoverflow.com/a/35213951/12224183)
    y = np.zeros(l, dtype=np.int8)  # [https://numpy.org/doc/stable/reference/arrays.scalars.html#sized-aliases](numpy.intc — Sized aliases)
    fX = X @ w + b
    y[fX >= 0] = 1
    y[fX < 0] = -1
    center_0, center_1 = np.mean(X[fX >= 0], axis=0), np.mean(X[fX < 0], axis=0)
    X[fX >= 0] += shrinkage * (center_0 - X[fX >= 0])
    X[fX < 0] += shrinkage * (center_1 - X[fX < 0])
    return X, y


def print_Xy(X, y):
    l, n = X.shape
    print('X:', X, 'y:', y)
    print('max(X): {}, min(X): {}, X.shape: {}'.format(
        np.max(X, axis=0), np.min(X, axis=0), X.shape)
    )
    print('number of +1 labels: {}, number of -1 labels:, y.shape: {}'.format(
        (l + np.sum(y)) // 2, (l - np.sum(y)) // 2, y.shape
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--dataset_size', type=int, default=10)
    parser.add_argument('--num_points', type=int, nargs='+', default=[100])
    parser.add_argument('--dimension', type=int, nargs='+', default=[2, 3])
    parser.add_argument('--w_min', type=float, default=-10)
    parser.add_argument('--w_max', type=float, default=10)
    parser.add_argument('--b_min', type=float, default=-10)
    parser.add_argument('--b_max', type=float, default=10)
    parser.add_argument('--shrinkage', type=float, nargs='+', default=[0.3, 0.5, 0.7])
    opts = parser.parse_args()

    np.random.seed(opts.seed)
    w_min, w_max = opts.w_min, opts.w_max
    b_min, b_max = opts.b_min, opts.b_max
    assert w_max >= w_min and b_max >= b_min
    for shrinkage in opts.shrinkage:
        assert 0 <= shrinkage <= 1
    for num_points in opts.num_points:
        data_dir = '{}_{}'.format(opts.data_dir, num_points)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        for dimension in opts.dimension:
            for shrinkage in opts.shrinkage:
                dataset = []
                for i in range(opts.dataset_size):
                    w = np.random.uniform(low=w_min, high=w_max, size=dimension)
                    b = np.random.uniform(low=b_min, high=b_max)
                    dataset.append(generate(l=num_points, w=w, b=b, shrinkage=shrinkage))
                filename = 'np{}_d{}_wmin{}_wmax{}_bmin{}_bmax{}_skg{}_sd{}.pkl'.format(
                    num_points, dimension, w_min, w_max, b_min, b_max, shrinkage, opts.seed
                )
                path = '{}/{}'.format(data_dir, filename)
                print('create {} data in {}'.format(opts.dataset_size, path))
                with open(path, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
