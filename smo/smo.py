import argparse
import numpy as np
import time
import os
import pickle
import json
import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate import generate, print_Xy


def smo(X, y, cache_max, epsilon=1e-3, tau=1e-12, verbose=False, verify=True):
    """

    @param X: instances in shape (l, n), where l is number of instances, n is the feature dimension
    @param y: labels in shape (l,)
    @param epsilon; stopping tolerance
    @return: alpha in shape (l,); number of iterations
    """
    assert X.ndim == 2 and y.ndim == 1
    l, n = X.shape
    assert l == len(y)
    cache_max = min(cache_max, l)
    Q_diag = np.square(y) * np.sum(np.square(X), axis=-1)  # (l)
    count = np.zeros(l, dtype=np.uint)
    Q_cache = np.zeros((cache_max, l), dtype=X.dtype)
    cache_len = 0
    l2c = dict()
    alpha = np.zeros(l, dtype=X.dtype)  # initialize alpha
    gradient = -1 * np.ones(l, dtype=X.dtype)  # initialize gradient
    iter = 0
    support_sum = 0
    logs = []
    start_time = time.time()
    while True:
        # select B, i.e. (i, j)
        i = -1  # select i
        G_max = -np.infty
        G_min = np.infty
        for t in range(l):
            if y[t] == 1 or (y[t] == -1 and alpha[t] > 0):  # y[t] == 1 or (y[t] == -1 and alpha[t] > 0)
                if -y[t] * gradient[t] >= G_max:
                    i = t
                    G_max = -y[t] * gradient[t]
        count[i] += 1
        if i not in l2c:
            if cache_len < cache_max:  # add
                Q_cache[cache_len] = y[i] * y * (X @ X[i])
                l2c[i] = l2c.setdefault(i, cache_len)
                cache_len += 1
            else:  # replace
                (bad_l_idx, bad_c_idx) = min(l2c.items(), key=lambda x: count[x[0]])
                Q_cache[bad_c_idx] = y[i] * y * (X @ X[i])
                del l2c[bad_l_idx]
                l2c[i] = bad_c_idx

        j = -1  # select j
        obj_min = np.infty
        for t in range(l):
            if (y[t] == 1 and alpha[t] > 0) or y[t] == -1:  # (y[t] == 1 and alpha[t] > 0) or y[t] == -1
                b = G_max + y[t] * gradient[t]
                if -y[t] * gradient[t] <= G_min:
                    G_min = -y[t] * gradient[t]
                if b > 0:
                    a = Q_diag[i] + Q_diag[t] - 2 * y[i] * y[t] * Q_cache[l2c[i], t]
                    if a <= 0:
                        a = tau
                    if -b * b / a <= obj_min:
                        j = t
                        obj_min = -b * b / a
        if G_max - G_min < epsilon:
            i, j = -1, -1
        iter += 1
        if verbose:
            sv_indices = np.where(alpha > 0)[0]  # support vector indices
            support_sum, support_sum_last = len(sv_indices), support_sum
            if support_sum != support_sum_last or j == -1:
                w_est = X[sv_indices].T @ (alpha[sv_indices] * y[sv_indices])
                b_est = np.mean(y[sv_indices][:, None] - X[sv_indices] @ w_est[:, None])
                log = {
                    'iter': iter, 'w_est': w_est, 'b_est': b_est, '(i,j)': (i, j), 'b': b, 'a': a,
                       'G_max-G_min': G_max - G_min, '(alpha>0).sum()': support_sum, 'time': time.time() - start_time
                }
                if verify:
                    # varify KKT conditions
                    fx = 1 - y * (X @ w_est + b_est)
                    kkt1 = fx[np.where(fx > 0)]
                    log['kkt1_max'], log['kkt1_num'] = np.max(kkt1, initial=0.0), len(kkt1)

                    kkt2 = -alpha[np.where(-alpha > 0)]
                    log['kkt2_max'], log['kkt2_num'] = np.max(kkt2, initial=0.0), len(kkt2)

                    axf = alpha * fx
                    kkt3 = abs(axf)[np.where(axf)]
                    log['kkt3_max'], log['kkt3_num'] = np.max(kkt3, initial=0.0), len(kkt3)

                    g_w = w_est - X.T @ (alpha * y)  # gradient on w
                    kkt4w = abs(g_w)[np.where(g_w)]
                    log['kkt4w_max'], log['kkt4w_num'] = np.max(kkt4w, initial=0.0), len(kkt4w)

                    g_b = np.inner(alpha, y)  # gradient on b
                    kkt4b = abs(g_b)
                    log['kkt4b_max'] = np.max(kkt4b, initial=0.0)
                logs.append(log)
                print(
                    'iter:', iter,
                    'w_est:', ', '.join(['{:.2f}'.format(_) for _ in w_est]),
                    'b_est: {:.2f}'.format(b_est),
                    '(i,j): ({},{})'.format(i, j),
                    'b: {:.3f}, a: {:.3f}'.format(b, a),
                    'G_max-G_min: {:.5f}'.format(G_max - G_min),
                    '(alpha>0).sum():', support_sum,
                    'α:', ', '.join(['{:.1f}'.format(_) for _ in alpha[sv_indices][:5]]),
                )
        if j == -1:
            if verbose:
                temp = np.where(count > 0)[0]
                return alpha, {
                    'count': sorted(zip(temp, count[temp]), key=lambda x: x[1], reverse=True),
                    'cache_len': cache_len,
                    'l2c': l2c,
                    'log': logs
                }
            else:
                return alpha, dict()

        # working set is (i, j)
        a = Q_diag[i] + Q_diag[j] - 2 * y[i] * y[j] * Q_cache[l2c[i], j]
        if a <= 0:
            a = tau
        b = -y[i] * gradient[i] + y[j] * gradient[j]

        # update alpha
        alpha_i_old, alpha_j_old = alpha[i], alpha[j]
        alpha[i] += y[i] * b / a
        alpha[j] -= y[j] * b / a

        # project alpha back to the feasible region
        alpha_sum = y[i] * alpha_i_old + y[j] * alpha_j_old
        if alpha[i] < 0:
            alpha[i] = 0
        alpha[j] = y[j] * (alpha_sum - y[i] * alpha[i])
        if alpha[j] < 0:
            alpha[j] = 0
        alpha[i] = y[i] * (alpha_sum - y[j] * alpha[j])

        # update gradient
        count[j] += 1
        if j not in l2c:
            if cache_len < cache_max:  # add
                Q_cache[cache_len] = y[j] * y * (X @ X[j])
                l2c[j] = l2c.setdefault(j, cache_len)
                cache_len += 1
            else:  # replace
                count_max = np.max(count)
                count[i] += count_max
                (bad_l_idx, bad_c_idx) = min(l2c.items(), key=lambda x: count[x[0]])
                Q_cache[bad_c_idx] = y[j] * y * (X @ X[j])
                del l2c[bad_l_idx]
                l2c[j] = bad_c_idx
                count[i] -= count_max  # avoid i to be deleted
        alpha_i_delta, alpha_j_delta = alpha[i] - alpha_i_old, alpha[j] - alpha_j_old
        gradient += Q_cache[l2c[i]] * alpha_i_delta + Q_cache[l2c[j]] * alpha_j_delta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--num_points', type=int, default=100)
    parser.add_argument('--dimension', type=int, default=2)
    parser.add_argument('--w', type=float, nargs='+')
    parser.add_argument('--b', type=float)
    parser.add_argument('--shrinkage', type=float, default=0.3)

    # smo parameters
    parser.add_argument('--epsilon', type=float, default=1e-3)
    parser.add_argument('--tau', type=float, default=1e-12)
    parser.add_argument('--cache_max', type=int, default=100)
    parser.add_argument('--verbose', default=True, action=argparse.BooleanOptionalAction)
    opts = parser.parse_args()
    print('opts:', opts)
    if opts.seed is not None:
        np.random.seed(opts.seed)
    epsilon, tau, cache_max, verbose = opts.epsilon, opts.tau, opts.cache_max, opts.verbose

    data_dir = opts.data_dir
    if data_dir:
        path_test = 'smo/results/dd{}_cm{}_eps{}_tau{}_vbs{}'.format(data_dir, cache_max, epsilon, tau, verbose)
        if not os.path.exists(path_test):
            os.makedirs(path_test)
        filenames = os.listdir(data_dir)
        for filename in tqdm.tqdm(filenames, desc='outer', position=0):
            path_load = '{}/{}'.format(data_dir, filename)
            with open(path_load, 'rb') as f:
                dataset = pickle.load(f)
            print('successfully load', path_load)
            filename = '.'.join(filename.split('.')[:-1])
            time_array = np.zeros(len(dataset))
            kkt1_array_max, kkt2_array_max, kkt3_array_max, kkt4w_array_max, kkt4b_array_max = [np.zeros(len(dataset)) for _ in range(5)]
            kkt1_array_num, kkt2_array_num, kkt3_array_num, kkt4w_array_num = [np.zeros(len(dataset)) for _ in range(4)]
            for i, data in tqdm.tqdm(enumerate(dataset), desc='inner', position=1, leave=False):
                X, y = data
                print_Xy(X, y)

                # solve by smo
                time_start = time.time()
                alpha, info = smo(np.array(X), np.array(y), cache_max, epsilon=epsilon, tau=tau, verbose=verbose)
                time_array[i] = time.time() - time_start

                # calculate support vectors and estimate w & b
                sv_indices = np.where(alpha > 0)[0]  # support vector indices
                w_est = X[sv_indices].T @ (alpha[sv_indices] * y[sv_indices])
                b_est = np.mean(y[sv_indices][:, None] - X[sv_indices] @ w_est[:, None])

                # varify KKT conditions
                fx = 1 - y * (X @ w_est + b_est)
                kkt1 = fx[np.where(fx > 0)]
                kkt1_array_max[i], kkt1_array_num[i] = np.max(kkt1, initial=0.0), len(kkt1)

                kkt2 = -alpha[np.where(-alpha > 0)]
                kkt2_array_max[i], kkt2_array_num[i] = np.max(kkt2, initial=0.0), len(kkt2)

                axf = alpha * fx
                kkt3 = abs(axf)[np.where(axf)]
                kkt3_array_max[i], kkt3_array_num[i] = np.max(kkt3, initial=0.0), len(kkt3)

                g_w = w_est - X.T @ (alpha * y)  # gradient on w
                kkt4w = abs(g_w)[np.where(g_w)]
                kkt4w_array_max[i], kkt4w_array_num[i] = np.max(kkt4w, initial=0.0), len(kkt4w)

                g_b = np.inner(alpha, y)  # gradient on b
                kkt4b = abs(g_b)
                kkt4b_array_max[i] = np.max(kkt4b, initial=0.0)

                if verbose:
                    with open('{}/{}_{}.pkl'.format(path_test, filename, i), 'wb') as f:
                        pickle.dump(info['log'], f, pickle.HIGHEST_PROTOCOL)

            result = {
                'time': (np.min(time_array), np.mean(time_array), np.max(time_array), np.std(time_array)),
                'kkt1_max': (np.min(kkt1_array_max), np.mean(kkt1_array_max), np.max(kkt1_array_max), np.std(kkt1_array_max)),
                'kkt1_num': (np.min(kkt1_array_num), np.mean(kkt1_array_num), np.max(kkt1_array_num), np.std(kkt1_array_num)),
                'kkt2_max': (np.min(kkt2_array_max), np.mean(kkt2_array_max), np.max(kkt2_array_max), np.std(kkt2_array_max)),
                'kkt2_num': (np.min(kkt2_array_num), np.mean(kkt2_array_num), np.max(kkt2_array_num), np.std(kkt2_array_num)),
                'kkt3_max': (np.min(kkt3_array_max), np.mean(kkt3_array_max), np.max(kkt3_array_max), np.std(kkt3_array_max)),
                'kkt3_num': (np.min(kkt3_array_num), np.mean(kkt3_array_num), np.max(kkt3_array_num), np.std(kkt3_array_num)),
                'kkt4w_max': (np.min(kkt4w_array_max), np.mean(kkt4w_array_max), np.max(kkt4w_array_max), np.std(kkt4w_array_max)),
                'kkt4w_num': (np.min(kkt4w_array_num), np.mean(kkt4w_array_num), np.max(kkt4w_array_num), np.std(kkt4w_array_num)),
                'kkt4b_max': (np.min(kkt4b_array_max), np.mean(kkt4b_array_max), np.max(kkt4b_array_max), np.std(kkt4b_array_max)),
            }
            with open('{}/{}.json'.format(path_test, filename), 'w') as f:
                json.dump(result, f, indent=4, sort_keys=False)
    else:
        # generate data
        l, w, b, shrinkage = opts.num_points, opts.w, opts.b, opts.shrinkage
        assert 0 <= shrinkage <= 1
        X, y = generate(l=l, w=w, b=b, shrinkage=shrinkage)
        print_Xy(X, y)

        # solve by smo
        time_start = time.time()
        alpha, info = smo(np.array(X), np.array(y), cache_max, epsilon=epsilon, tau=tau, verbose=verbose)
        time_solve = time.time() - time_start
        print('solve time:', time_solve, end='\n\n')

        # calculate support vectors and estimate w & b
        sv_indices = np.where(alpha > 0)[0]  # support vector indices
        w_est = X[sv_indices].T @ (alpha[sv_indices] * y[sv_indices])
        b_est = np.mean(y[sv_indices][:, None] - X[sv_indices] @ w_est[:, None])
        print('support vector indexes:', sv_indices)
        print(
            'input w:', ', '.join(['{:.3f}'.format(_) for _ in w]),
            'estimated w:', ', '.join(['{:.3f}'.format(_) for _ in w_est]),
        )
        print('intput b:', b, 'estimated b:', b_est, end='\n\n')

        # varify KKT conditions
        fx = 1 - y * (X @ w_est + b_est)
        fge0_idx = np.where(fx > 0)[0]
        fge0_val = fx[fge0_idx]
        print('KKT1 (primal constraints), f>0 (index, value) pairs:', list(zip(fge0_idx, fge0_val)))

        age0_idx = np.where(alpha < 0)[0]
        age0_val = alpha[age0_idx]
        print('KKT2 (dual constraints), alpha>0 (index, value) pairs:', list(zip(age0_idx, age0_val)))

        axf = alpha * fx
        axfne0_idx = np.where(axf != 0)[0]
        axfne0_val = axf[axfne0_idx]
        print('KKT3 (complementary slackness), alpha*fx≠0 (index, value) pairs:', list(zip(axfne0_idx, axfne0_val)))

        g_w = w_est - X.T @ (alpha * y)  # gradient on w
        g_b = np.inner(alpha, y)  # gradient on b
        print('KKT4 (gradient of Lagrangian w.r.t. x vanishes), gradient on w: {}, gradient on b: {}'.format(g_w, g_b))
