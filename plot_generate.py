import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import sklearn
import os
from generate import generate


if not os.path.exists('figures'):
    os.makedirs('figures')
l = 1000

for shrinkage in [0, 0.2, 0.4, 0.6, 0.8, 1]:
    w = np.random.uniform(low=-10, high=10, size=2)
    b = np.random.uniform(low=-10, high=10)
    X, y = generate(l=l, w=w, b=b, shrinkage=shrinkage)

    # scikit-learn SVM examples:
    # 1. [sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
    # 2. [sklearn.svm.LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
    # note that if using make_pipeline(StandardScaler(), SVC(gamma='auto')), the coef_ and intercept_ is somehow standardized
    print('*' * 15, 'START sklearn.svm.SVC', '*' * 15)
    clf = sklearn.svm.SVC(kernel='linear', C=1000, verbose=True)
    clf.fit(X, y)
    for key in dir(clf):
        if key[0] != '_' and key[-1] == '_':
            print('{}: {}'.format(key, getattr(clf, key)))
    coef_, intercept_ = clf.coef_[0], clf.intercept_[0]
    support_vectors_ = clf.support_vectors_
    print('*' * 15, 'END sklearn.svm.SVC', '*' * 15)

    print('*' * 15, 'START sklearn.svm.LinearSVC', '*' * 15)
    clf = sklearn.svm.LinearSVC(dual=True, C=1000, tol=1e-5, verbose=True)
    clf.fit(X, y)
    for key in dir(clf):
        if key[0] != '_' and key[-1] == '_':
            print('{}: {}'.format(key, getattr(clf, key)))
    print('*' * 15, 'END sklearn.svm.LinearSVC', '*' * 15)
    if True and len(w) == 2:
        plt.rcParams['figure.figsize'] = (4, 4)  # [How do I change the size of figures drawn with Matplotlib?](https://stackoverflow.com/a/41717533/12224183)
        # linear SVM plot examples by using function DecisionBoundaryDisplay.from_estimator
        # 1. [Plot the support vectors in LinearSVC](https://scikit-learn.org/stable/auto_examples/svm/plot_linearsvc_support_vectors.html)
        # 2. [Plot classification boundaries with different SVM Kernels](https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html#sphx-glr-auto-examples-svm-plot-svm-kernels-py)
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='C0', alpha=0.5)
        plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='C1', alpha=0.5)
        plt.scatter(support_vectors_[:, 0], support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')
        minX0, maxX0 = np.min(X[:, 0]), np.max(X[:, 0])
        minX1, maxX1 = np.min(X[:, 1]), np.max(X[:, 1])
        temp = np.linspace(start=minX0, stop=maxX0, num=50)
        plt.plot(temp, (-w[0] * temp - b) / w[1], color='C2')
        plt.plot(temp, (-coef_[0] * temp - intercept_) / coef_[1], color='C3')  # [Decision boundary calculation in SVM](https://stackoverflow.com/a/43574997/12224183)
        plt.xlim(minX0, maxX0)
        plt.ylim(minX1, maxX1)
        figure_name = 'l:{:.0e},w:['.format(l) + ''.join(['{:.1f}'.format(_) for _ in w]) + '],b:{:.1f},'.format(b) + 's:{}'.format(shrinkage)
        plt.title(figure_name)
        plt.savefig('figures/l{}-s{}.png'.format(l, shrinkage), bbox_inches='tight')
        plt.clf()
    print('true [w; b]: [{}; {}], estimated [w; b]: [{}; {}]'.format(
        w, b, coef_, intercept_
    ))
    print('support vector:', support_vectors_)
