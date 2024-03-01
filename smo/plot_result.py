import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.legend_handler import HandlerTuple
import sklearn
from sklearn.svm import SVC
import numpy as np
import time


if not os.path.exists('smo/figures'):
    os.makedirs('smo/figures')

# load result of self implemented svm
num_points = 100
data_dir = 'data_{}'.format(num_points)
filename = 'np{}_d3_wmin-10_wmax10_bmin-10_bmax10_skg0.3_sd1234'.format(num_points)
run_name = 'dd{}_cm100_eps0.001_tau1e-12_vbsTrue'.format(data_dir)
C = 1000
for data_idx in range(10):
    with open('smo/results/{}/{}_{}.pkl'.format(run_name, filename, data_idx), 'rb') as f:
        logs = pickle.load(f)

    # solve by scikit learn LIBSVM SMO
    start_time = time.time()
    with open('{}/{}.pkl'.format(data_dir, filename), 'rb') as f:
        data = pickle.load(f)
        X, y = data[data_idx]
    clf = sklearn.svm.SVC(kernel='linear', C=C, verbose=True)
    clf.fit(X, y)
    for key in dir(clf):
        if key[0] != '_' and key[-1] == '_':
            print('{}: {}'.format(key, getattr(clf, key)))
    coef_, intercept_ = clf.coef_[0], clf.intercept_[0]
    dual_coef_ = clf.dual_coef_  # {ndarray: (1, n)}
    support_ = clf.support_  # {ndarray: (n,)}
    support_vectors_ = clf.support_vectors_  # {ndarray: (n, d)}
    libsvm_time = time.time() - start_time

    # verify KKT for scikit learn LIBSVM SMO
    alpha = np.zeros(len(X))
    alpha[support_] = dual_coef_

    fx = 1 - y * (X @ coef_.T + intercept_)
    kkt1 = fx[np.where(fx > 0)]
    kkt1_max, kkt1_num = np.max(kkt1, initial=0.0), len(kkt1)

    kkt2 = -alpha[np.where(-alpha > 0)]
    kkt2_max, kkt2_num = np.max(kkt2, initial=0.0), len(kkt2)

    axf = alpha * fx
    kkt3 = abs(axf)[np.where(axf)]
    kkt3_max, kkt3_num = np.max(kkt3, initial=0.0), len(kkt3)

    g_w = coef_.T - X.T @ (alpha * y)  # gradient on w
    kkt4w = abs(g_w)[np.where(g_w)]
    kkt4w_max, kkt4w_num = np.max(kkt4w, initial=0.0), len(kkt4w)

    g_b = np.inner(alpha, y)  # gradient on b
    kkt4b = abs(g_b)
    kkt4b_max = np.max(kkt4b, initial=0.0)

    # plot, [Plots with different scales](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html)
    step = np.array([log['iter'] for i, log in enumerate(logs) if i != 0])
    kkt1_array_max = np.array([log['kkt1_max'] for i, log in enumerate(logs) if i != 0])
    kkt1_array_num = np.array([log['kkt1_num'] for i, log in enumerate(logs) if i != 0])
    kkt2_array_max = np.array([log['kkt2_max'] for i, log in enumerate(logs) if i != 0])
    kkt2_array_num = np.array([log['kkt2_num'] for i, log in enumerate(logs) if i != 0])
    kkt3_array_max = np.array([log['kkt3_max'] for i, log in enumerate(logs) if i != 0])
    kkt3_array_num = np.array([log['kkt3_num'] for i, log in enumerate(logs) if i != 0])
    kkt4w_array_max = np.array([log['kkt4w_max'] for i, log in enumerate(logs) if i != 0])
    kkt4w_array_num = np.array([log['kkt4w_num'] for i, log in enumerate(logs) if i != 0])
    kkt4b_array_max = np.array([log['kkt4b_max'] for i, log in enumerate(logs) if i != 0])
    my_time = logs[-1]['time']

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel('Number of steps')
    color1, color2, color3, color4, color5 = 'C0', 'C1', 'C2', 'C3', 'C4'

    # max violation
    ax.set_title(
        'Max violation\nrun name: {}\nfilename: {}\ndata index: {}\nsolved by mySMO/LIBSVM(C={}) in {:.2f}s/{:.2f}s'.format(
            run_name, filename, data_idx, C, my_time, libsvm_time)
    )
    ax.set_ylabel('Max violation')
    ax.set_yscale('log')
    p1, = ax.plot(step, kkt1_array_max, color=color1)
    p1_ = ax.axhline(y=kkt1_max, color=color1, linestyle=':')
    p2, = ax.plot(step, kkt2_array_max, color=color2)
    p2_ = ax.axhline(y=kkt2_max, color=color2, linestyle=':')
    p3, = ax.plot(step, kkt3_array_max, color=color3)
    p3_ = ax.axhline(y=kkt3_max, color=color3, linestyle=':')
    p4w, = ax.plot(step, kkt4w_array_max, color=color4)
    p4w_ = ax.axhline(y=kkt4w_max, color=color4, linestyle=':')
    p4b, = ax.plot(step, kkt4b_array_max, color=color5)
    p4b_ = ax.axhline(y=kkt4b_max, color=color5, linestyle=':')
    ax.add_artist(ax.legend(
        [(p1, p2, p3, p4w, p4b), (p1_, p2_, p3_, p4w_, p4b_)], ['implemented SMO', 'LIBSVM SMO'],
        handler_map={tuple: HandlerTuple(ndivide=None)}, loc='center left'
    ))  # [How to make two markers share the same label in the legend](https://stackoverflow.com/a/54980605)
    ax.add_artist(ax.legend(
        [(p1, p1_), (p2, p2_), (p3, p3_), (p4w, p4w_), (p4b, p4b_)],
        ['Primal constraints', 'Dual constraints', 'Complementary slackness', 'Gradient on w vanishes',
         'Gradient on b vanishes'],
        handler_map={tuple: HandlerTuple(ndivide=None)}, loc='center right'
    ))  # [matplotlib: 2 different legends on same graph](https://stackoverflow.com/a/12762069)
    plt.savefig('smo/figures/{}_{}_max.png'.format(filename, data_idx), bbox_inches='tight')
    plt.cla()

    # number of violations
    ax.set_title(
        'Number of violations\nrun name: {}\nfilename: {}\ndata index: {}\nsolved by mySMO/LIBSVM(C={}) in {:.2f}s/{:.2f}s'.format(
            run_name, filename, data_idx, C, my_time, libsvm_time)
    )
    ax.set_ylabel('Number of violations')
    ax.yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))
    p1, = ax.plot(step, kkt1_array_num / num_points, color=color1)
    p1_ = ax.axhline(y=kkt1_num / num_points, color=color1, linestyle=':')
    p2, = ax.plot(step, kkt2_array_num / num_points, color=color2)
    p2_ = ax.axhline(y=kkt2_num / num_points, color=color2, linestyle=':')
    p3, = ax.plot(step, kkt3_array_num / num_points, color=color3)
    p3_ = ax.axhline(y=kkt3_num / num_points, color=color3, linestyle=':')
    p4w, = ax.plot(step, kkt4w_array_num / num_points, color=color4)
    p4w_ = ax.axhline(y=kkt4w_num / num_points, color=color4, linestyle=':')
    ax.add_artist(ax.legend(
        [(p1, p2, p3, p4w), (p1_, p2_, p3_, p4w_)], ['implemented SMO', 'LIBSVM SMO'],
        handler_map={tuple: HandlerTuple(ndivide=None)}, loc='center left'
    ))  # [How to make two markers share the same label in the legend](https://stackoverflow.com/a/54980605)
    ax.add_artist(ax.legend(
        [(p1, p1_), (p2, p2_), (p3, p3_), (p4w, p4w_)],
        ['Primal constraints', 'Dual constraints', 'Complementary slackness', 'Gradient on w vanishes'],
        handler_map={tuple: HandlerTuple(ndivide=None)}, loc='center right'
    ))  # [matplotlib: 2 different legends on same graph](https://stackoverflow.com/a/12762069)
    plt.savefig('smo/figures/{}_{}_num.png'.format(filename, data_idx), bbox_inches='tight')
    plt.cla()

