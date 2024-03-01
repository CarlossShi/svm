import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import sklearn
import math
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate import generate


def PDCD(X, y, delta=0.1, epsilon=1e-3, epsilon_bar=1e-15, initB_bar=256, maxB_bar=4096, tol=1e-4, verbose=False):
    """
    Parallel Dual Coordinate Descent proposed in [Parallel Dual Coordinate Descent Method for Large-scale Linear Classification in Multi-core Environments](https://dl.acm.org/doi/10.1145/2939672.2939826)
    @param X: instances in shape (l, n), where l is number of instances, n is the feature dimension
    @param y: labels in shape (l,)
    @param delta: tolerance for updating w, "Note that we need δ ∈ (0, 1) to ensure from the stopping condition (10) that at each outer iteration at least one αi is updated."
    @param epsilon: tolerance for updating w and ε_1, termination condition
    @param epsilon_bar: tolerance for reject too small updates, "Note that Algorithm 4 has another parameter ε_bar ≪ ε, so we also need ε_bar → 0 as well."
    @param initB_bar:  "Because in general 0 < |B| < init B_bar, |B_bar| is seldom changed in practice. Hence our rule mainly serves as a safeguard."
    @param maxB_bar: "If |B| = 0, to get some elements in B for CD updates, we should enlarge B_bar. In contrast, if too many elements are included in B, we should reduce the size of B_bar."
    @return: alpha in shape (l,); number of iterations
    """
    assert X.ndim == 2 and y.ndim == 1
    l, n = X.shape
    assert l == len(y)
    epsilon_1 = 0.1
    alpha = np.zeros(l)  # L1
    support_sum = support_sum_last = 0
    w = X.T @ (alpha * y)  # L1
    primal_obj = primal_obj_last = np.infty
    M_bar, m_bar = np.infty, -np.infty  # L3
    A = np.ones(l, dtype=bool)  # L3
    A_sum = l
    nowB_bar = initB_bar  # L4
    iter = 0
    while True:
        M, m, A_bar, t_bar = -np.infty, np.infty, A.copy(), 0  # L6
        A_bar_sum = A_sum
        while A_bar_sum:  # [Python's sum vs. NumPy's numpy.sum](https://stackoverflow.com/a/49908528/12224183)
            B_bar_len = min(math.ceil(nowB_bar), A_bar_sum)
            B_bar = np.random.choice(a=np.where(A_bar)[0], size=B_bar_len, replace=False)  # L8, "We select a set B_bar and calculate the corresponding gradient values in parallel."
            Q = (X[B_bar] * X[B_bar]).sum(axis=1)
            A_bar[B_bar] = 0  # L9
            A_bar_sum -= B_bar_len
            nabla = y[B_bar] * (X[B_bar] @ w) - np.ones(len(B_bar))  # L10
            B = np.zeros(len(B_bar), dtype=bool)  # L11, "We then get a much smaller set B ⊂ B_bar and update α_B."
            B_sum = 0
            for i_b, i in enumerate(B_bar):  # L12
                PG, G = 0, nabla[i_b]  # L13
                if G < 0 or (alpha[i] > 0 and G > 0):  # L14
                    PG = G  # L15
                elif alpha[i] == 0 and G > M_bar:  # L16
                    A[i] = 0  # L17
                    A_sum -= 1
                M, m = max(M, PG), min(m, PG)  # L18
                if abs(PG) >= delta * epsilon_1:  # L19
                    B[i_b] = 1  # L20
                    B_sum += 1
            if not B_sum:  # L21
                nowB_bar = min(1.5 * nowB_bar, maxB_bar)  # L22
            elif B_sum >= initB_bar:  # L23
                nowB_bar = nowB_bar / 2  # L24
            for i_b in np.where(B)[0]:
                i = B_bar[i_b]  # L25
                G = y[i] * np.inner(w, X[i]) - 1  # L26
                d = max(alpha[i] - G / Q[i_b], 0) - alpha[i]  # L27
                if abs(d) >= epsilon_bar:  # L28
                    alpha[i] += d  # L29
                    w += d * y[i] * X[i]  # L30
                    t_bar += 1  # L31

        primal_obj, primal_obj_last = 0.5 * np.inner(w, w), primal_obj
        support_sum, support_sum_last = (alpha > 0).sum(), support_sum
        iter += 1
        if abs(primal_obj - primal_obj_last) < tol:
            return alpha, iter
        if verbose and support_sum != support_sum_last:
            print(
                'iter:', iter, 'p_obj: {:.3f}'.format(primal_obj),
                'α:', ', '.join(['{:.1f}'.format(_) for _ in alpha[alpha > 0][:5]]),
                '(alpha>0).sum(): {}, A.sum(): {}, nowB_bar: {:.0f}, log10(M-m): {:.1f}, log10(ε_1): {:.1f}, t_bar: {}'.format(
                    support_sum, A_sum, nowB_bar, -np.infty if M == m else math.log10(M - m), math.log10(epsilon_1), t_bar
                ),
                'w:', ', '.join(['{:.1f}'.format(_) for _ in w])
            )

        if M - m <= epsilon_1 or t_bar == 0:  # L32
            if A_sum == l and epsilon_1 <= epsilon:  # L33
                return alpha, iter  # L34
            else:  # L35
                A, M_bar, m_bar = np.ones(l, dtype=bool), np.infty, -np.infty  # L36
                A_sum = l
                epsilon_1 = max(0.1 * epsilon_1, epsilon)  # L37
        M_bar = np.infty if M <= 0 else M  # L38-41
        m_bar = -np.infty if m >= 0 else m  # L42-45


if __name__ == '__main__':
    l = 100
    w = np.array([2, 1])
    b = 10
    X, y = generate(l=l, w=w, b=b, shrinkage=0.8)

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
    if False and len(w) == 2:
        # linear SVM plot examples by using function DecisionBoundaryDisplay.from_estimator
        # 1. [Plot the support vectors in LinearSVC](https://scikit-learn.org/stable/auto_examples/svm/plot_linearsvc_support_vectors.html)
        # 2. [Plot classification boundaries with different SVM Kernels](https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html#sphx-glr-auto-examples-svm-plot-svm-kernels-py)
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='C0')
        plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='C1')
        plt.scatter(support_vectors_[:, 0], support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')
        minX0, maxX0 = np.min(X[:, 0]), np.max(X[:, 0])
        minX1, maxX1 = np.min(X[:, 1]), np.max(X[:, 1])
        temp = np.linspace(start=minX0, stop=maxX0, num=50)
        plt.plot(temp, (-w[0] * temp - b) / w[1], color='C2')
        plt.plot(temp, (-coef_[0] * temp - intercept_) / coef_[1], color='C3')  # [Decision boundary calculation in SVM](https://stackoverflow.com/a/43574997/12224183)
        plt.xlim(minX0, maxX0)
        plt.ylim(minX1, maxX1)
        plt.show()
    print('true [w; b]: [{}; {}], estimated [w; b]: [{}; {}]'.format(
        w, b, coef_, intercept_
    ))
    print('support vector:', support_vectors_)

    X_augmented = np.concatenate((X, np.ones((len(X), 1))), axis=1)
    alpha, iter = PDCD(X_augmented, y, initB_bar=256, maxB_bar=4096, verbose=True)
    w_pdcd = X_augmented.T @ (alpha * y)
    print(w_pdcd)