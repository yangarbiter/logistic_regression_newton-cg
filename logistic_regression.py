
import numpy as np
import scipy.sparse
#from sklearn.datasets import load_svmlight_file
from svmlight_loader import load_svmlight_file

def ywx(w, X, y):
    return (-1 * (X * w.T).T * y).reshape(1, -1)

def f(w, X, y, C):
    e = np.log(np.exp(ywx(w, X, y)) + 1)
    return 0.5 * np.dot(w, w.T) + C * np.sum(e)

def gf(w, X, y, C):
    e = 1. / (np.exp(ywx(w, X, y)) + 1) - 1
    return (w + C * e * (scipy.sparse.diags(y, 0) * X)).reshape(1, -1)

def fa(wx, dx, y, C):
    e = np.log(np.exp((-1 * (wx + dx).T * y).reshape(1, -1)) + 1)
    return C * np.sum(e)

def fb(wx, y, C):
    e = np.log(np.exp((-1 * wx.T * y).reshape(1, -1)) + 1)
    return C * np.sum(e)


def gradient(X, y):
    C = 0.1
    eps = 0.01

    beta = 0.5
    eta = 0.01
    w_k = np.zeros(np.shape(X)[1]).reshape(1, -1)
    s_0 = gf(w_k, X, y, C)
    s_k = -1 * s_0
    print np.shape(w_k + s_k)
    ite = 1

    inner = lambda x: np.dot(x, x.T)
    for k in range(1000):
        alpha = 1.

        #while f(w_k + alpha * s_k, X, y, C)[0][0] > f(w_k, X, y, C)[0][0] + eta * alpha * np.dot(-s_k.T, s_k)[0][0]:
        #    print f(w_k + alpha * s_k, X, y, C)
        #    alpha = alpha * beta

        wx = X * w_k.T
        dx = (X * s_k.T)
        ori_f = (0.5*inner(w_k) + C * np.sum(np.log1p(np.exp((-1 * wx.T * y)))))[0][0]

        # line search
        updated_f = (0.5*inner(w_k + alpha*s_k) + C * np.sum(np.log1p(np.exp((-1 * (wx + alpha*dx).T * y)))))[0][0]
        s_k2 = -inner(s_k)[0][0]
        while updated_f > ori_f + eta * alpha * s_k2:
            alpha = alpha * beta
            updated_f = (0.5*inner(w_k + alpha*s_k) + C * np.sum(np.log1p(np.exp((-1 * (wx + alpha*dx).T * y)))))[0][0]

        print 'iter %2d f %.6e |g| %.6e CG %2d step_size %f' % (ite, ori_f, np.linalg.norm(s_k), 0, alpha)

        w_k = w_k + alpha * s_k
        s_k = -1 * gf(w_k, X, y, C)

        ite += 1

        if np.linalg.norm(s_0) * eps > np.linalg.norm(s_k):
            break

    print 'iter %2d f %.6e |g| %.6e CG %2d step_size %f' % (ite, updated_f, np.linalg.norm(s_k), 0, alpha)

def newton_cg(X, y):
    C = 0.1
    eps = 0.01
    eta = 0.01
    xi = 0.1
    w_k = np.zeros((1, np.shape(X)[1]))
    s_0 = gf(w_k, X, y, C)
    ite = 1
    while True:

        e = np.exp(ywx(w_k, X, y))
        D = scipy.sparse.diags((e / (1. + e)**2).reshape(-1), 0)

        grad_f = gf(w_k, X, y, C)

        hfp = lambda si: si + C * si * X.T * D * X
        r = -gf(w_k, X, y, C)
        d = r
        s = np.zeros((1, np.shape(X)[1]))
        cg_iter = 0
        while True:
            if np.linalg.norm(r) <= xi * np.linalg.norm(grad_f):
                break
            hessian_product = hfp(d)
            norm_r = np.dot(r, r.T)[0][0]

            alpha_i = (norm_r / (np.dot(d, hessian_product.T)))[0][0]
            si1 = s + alpha_i * d
            ri1 = r - alpha_i * hessian_product
            beta = np.dot(ri1, ri1.T)[0][0] /  norm_r
            d = ri1 + beta*d
            r = ri1
            s = si1
            cg_iter += 1

        alpha = 1.
        s_k = s

        #while f(w_k + alpha * s_k, X, y, C)[0][0] > f(w_k, X, y, C)[0][0] + eta * alpha * np.dot(gf(w_k, X, y, C), s_k.T)[0][0]:
        #    #print f(w_k + alpha * s_k, X, y, C)
        #    alpha = alpha * beta

        inner = lambda x: np.dot(x, x.T)
        wx = X * w_k.T
        dx = (X * s_k.T)
        ori_f = (0.5*inner(w_k) + C * np.sum(np.log1p(np.exp((-1 * wx.T * y)))))[0][0]

        print -1 * (wx + alpha*dx).T * y
        updated_f = (0.5*inner(w_k + alpha*s_k) + C * np.sum(np.log1p(np.exp((-1 * (wx + alpha*dx).T * y)))))[0][0]
        s_k2 = np.dot(grad_f, s_k.T)[0][0]
        while updated_f > ori_f + eta * alpha * s_k2:
            alpha = alpha * beta
            updated_f = (0.5*inner(w_k + alpha*s_k) + C * np.sum(np.log1p(np.exp((-1 * (wx + alpha*dx).T * y)))))[0][0]

        print 'iter %2d f %.6e |g| %.6e CG %2d step_size %f' % (ite, ori_f, np.linalg.norm(grad_f) ,cg_iter, alpha)

        w_k = w_k + alpha * s_k
        ite += 1

        if np.linalg.norm(s_0) * eps > np.linalg.norm(grad_f):
            break

    print 'iter %2d f %.6e |g| %.6e CG %2d step_size %f' % (ite, updated_f, np.linalg.norm(gf(w_k, X, y, C)) ,cg_iter, alpha)

def main():
    #X, y = load_svmlight_file('./heart_scale')
    #X, y = load_svmlight_file('./large.dat')
    X, y = load_svmlight_file('./kddb')
    y[y==0] = -1
    gradient(X, y)
    #newton_cg(X, y)



if __name__ == '__main__':
    main()
