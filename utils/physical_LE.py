import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import rascaline

from scipy.special import spherical_jn as j_l
from scipy.special import spherical_in as i_l
from scipy.integrate import quadrature

from .spherical_bessel_zeros import Jn_zeros, get_laplacian_eigenvalues
from .LE_cutoff import get_LE_cutoff


def initialize_physical_LE(a, rs, E_max, r0, rnn):

    l_big = 0 if rs else 50
    n_big = 50

    pad_l = 5
    pad_n = 15

    E_nl = get_laplacian_eigenvalues(n_big, l_big, cost_trade_off=False)
    if rs:
        E_n0 = E_nl[:, 0]
    n_max, l_max = get_LE_cutoff(E_nl, E_max, rs)

    # Increase numbers with respect to LE cutoff:
    if rs:
        n_max_l = [n_max+pad_n]
    else:
        n_max_l = []
        for l in range(l_max+1):
            n_max_l.append(np.where(E_nl[:, l] <= E_max)[0][-1] + 1 + pad_n)
        for l in range(l_max+1, l_max+pad_l+1):
            n_max_l.append(pad_n)
        l_max = l_max + pad_l
    n_max = n_max + pad_n
    print(n_max_l)

    alpha = 1.0/r0

    z_ln = Jn_zeros(l_max+1, n_max)
    z_nl = z_ln.T

    def R_nl(n, l, r):
        return j_l(l, z_nl[n, l]*r/1.0)

    def dRnl_dr(n, l, r):
        return z_nl[n, l]/1.0*j_l(l, z_nl[n, l]*r/1.0, derivative=True)

    def d2Rnl_dr2(n, l, r):
        return (z_nl[n, l]/1.0)**2 * (
            l*(j_l(l, z_nl[n, l]*r/1.0, derivative=True)/(z_nl[n, l]*r/1.0) - j_l(l, z_nl[n, l]*r/1.0)/(z_nl[n, l]*r/1.0)**2) - j_l(l+1, z_nl[n, l]*r/1.0, derivative=True)
        )

    def N_nl(n, l):
        # Normalization factor for LE basis functions
        def function_to_integrate_to_get_normalization_factor(x):
            return j_l(l, x)**2 * x**2
        integral, _ = sp.integrate.quadrature(function_to_integrate_to_get_normalization_factor, 0.0, z_nl[n, l], maxiter=100)
        return ((1.0/z_nl[n, l])**3 * integral)**(-0.5)

    precomputed_N_nl = np.zeros((n_max, l_max+1))
    for n in range(n_max):
        for l in range(l_max+1):
            precomputed_N_nl[n, l] = N_nl(n, l)

    def get_LE_function(n, l, r):
        R = R_nl(n, l, r)
        return precomputed_N_nl[n, l]*R

    def get_LE_function_derivative(n, l, r):
        dR = dRnl_dr(n, l, r)
        return precomputed_N_nl[n, l]*dR

    def get_LE_function_second_derivative(n, l, r):
        d2R = d2Rnl_dr2(n, l, r)
        return precomputed_N_nl[n, l]*d2R

    # normalization check:
    def what(x):
        return get_LE_function(0, 0, x)**2 * x**2
    assert np.abs(1.0-sp.integrate.quadrature(what, 0.0, 1.0)[0]) < 1e-6
    def what(x):
        return get_LE_function(0, 0, x)*get_LE_function(4, 0, x) * x**2
    assert np.abs(sp.integrate.quadrature(what, 0.0, 1.0)[0]) < 1e-6

    def b(l, n, x):
        return get_LE_function(n, l, x)

    def db(l, n, x):
        return get_LE_function_derivative(n, l, x)
                
    def d2b(l, n, x):
        return get_LE_function_second_derivative(n, l, x)

    S = []
    for l in range(l_max+1):
        S_l = np.zeros((n_max_l[l], n_max_l[l]))
        for m in range(n_max_l[l]):
            for n in range(n_max_l[l]):
                Smn = sp.integrate.quadrature(
                    lambda x: b(l, m, x)*b(l, n, x)*(np.log(1.0-x))**2/((1.0-x)*alpha**3), 
                    0.0, 
                    1.0,
                    maxiter = 500
                )[0]
                S_l[m, n] = Smn
        # print(S_l)
        print(l)
        S.append(S_l)

    H = []
    for l in range(l_max+1):
        H_l = np.zeros((n_max_l[l], n_max_l[l]))
        for m in range(n_max_l[l]):
            for n in range(n_max_l[l]):
                Hmn = sp.integrate.quadrature(
                    lambda x: b(l, m, x)*(-d2b(l, n, x)-(2/x)*db(l, n, x)+(1/x**2)*l*(l+1)*b(l, n, x))*(np.log(1.0-x))**2/((1.0-x)*alpha**3), 
                    0.0, 
                    1.0,
                    maxiter = 500
                )[0]
                H_l[m, n] = Hmn
        # print(H_l)
        print(l)
        H.append(H_l)

    coeffs = []
    new_E_ln = []
    for l in range(l_max+1):
        eva, eve = sp.linalg.eigh(H[l], S[l])
        new_E_ln.append(eva)
        coeffs.append(eve)
    for l in range(l_max+1):
        print(new_E_ln[l])

    def function_for_splining(n, l, r):
        ret = np.zeros_like(r)
        for m in range(n_max_l[l]):
            ret += coeffs[l][m, n]*b(l, m, 1-np.exp(-r/r0))
        return ret

    def function_for_splining_derivative(n, l, r):
        ret = np.zeros_like(r)
        for m in range(n_max_l[l]):
            ret += coeffs[l][m, n]*db(l, m, 1-np.exp(-r/r0))*np.exp(-r/r0)/r0
        return ret

    if rs:
        E_nl = new_E_ln[0]
    else:
        E_nl = np.zeros((n_max, l_max+1))
        for l in range(l_max+1):
            for n in range(n_max):
                try:
                    E_nl[n, l] = new_E_ln[l][n]
                except:
                    E_nl[n, l] = 1e30
    n_max, l_max = get_LE_cutoff(E_nl, E_max, rs)
    if not rs:
        if l_max == E_nl.shape[1] - 1:
            raise ValueError("l pad is too low")

    spline_points = rascaline.generate_splines(
        function_for_splining,
        function_for_splining_derivative,
        n_max,
        l_max,
        a,
        requested_accuracy = 1e-6
    )
    print("Number of spline points:", len(spline_points))

    import matplotlib.pyplot as plt
    r = np.linspace(0.1, a-0.001, 1000)
    plt.plot(r, function_for_splining(0, 0, r), label=str(rs))
    plt.plot([0.0, a], [0.0, 0.0], "black")
    plt.xlim(0.0, a)
    plt.legend()
    plt.savefig("radial-real.pdf")

    return n_max, l_max, E_nl, spline_points
