import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import rascaline
import rascaline.torch
import multiprocessing

from scipy.special import spherical_jn as j_l
from scipy.special import spherical_in as i_l
from scipy.integrate import quadrature

from .spherical_bessel_zeros import Jn_zeros, get_laplacian_eigenvalues
from .LE_cutoff import get_LE_cutoff


def R_nl(n, l, r, z_nl):
    return j_l(l, z_nl[n, l]*r/1.0)

def dRnl_dr(n, l, r, z_nl):
    return z_nl[n, l]/1.0*j_l(l, z_nl[n, l]*r/1.0, derivative=True)

def d2Rnl_dr2(n, l, r, z_nl):
    return (z_nl[n, l]/1.0)**2 * (
        l*(j_l(l, z_nl[n, l]*r/1.0, derivative=True)/(z_nl[n, l]*r/1.0) - j_l(l, z_nl[n, l]*r/1.0)/(z_nl[n, l]*r/1.0)**2) - j_l(l+1, z_nl[n, l]*r/1.0, derivative=True)
    )

def N_nl(n, l, z_nl):
    # Normalization factor for LE basis functions
    def function_to_integrate_to_get_normalization_factor(x):
        return j_l(l, x)**2 * x**2
    integral, _ = sp.integrate.quadrature(function_to_integrate_to_get_normalization_factor, 0.0, z_nl[n, l], maxiter=500)
    return ((1.0/z_nl[n, l])**3 * integral)**(-0.5)

def get_LE_function(n, l, r, z_nl, precomputed_N_nl):
    R = R_nl(n, l, r, z_nl)
    return precomputed_N_nl[n, l]*R

def get_LE_function_derivative(n, l, r, z_nl, precomputed_N_nl):
    dR = dRnl_dr(n, l, r, z_nl)
    return precomputed_N_nl[n, l]*dR

def get_LE_function_second_derivative(n, l, r, z_nl, precomputed_N_nl):
    d2R = d2Rnl_dr2(n, l, r, z_nl)
    return precomputed_N_nl[n, l]*d2R

def b(l, n, x, z_nl, precomputed_N_nl):
    return get_LE_function(n, l, x, z_nl, precomputed_N_nl)

def db(l, n, x, z_nl, precomputed_N_nl):
    return get_LE_function_derivative(n, l, x, z_nl, precomputed_N_nl)
            
def d2b(l, n, x, z_nl, precomputed_N_nl):
    return get_LE_function_second_derivative(n, l, x, z_nl, precomputed_N_nl)

def integrate_S(l, m, n, alpha, z_nl, precomputed_N_nl):
    return sp.integrate.quadrature(
        lambda x: b(l, m, x, z_nl, precomputed_N_nl)*b(l, n, x, z_nl, precomputed_N_nl)*(np.log(1.0-x))**2/((1.0-x)*alpha**3), 
        0.0, 
        1.0,
        maxiter = 500
    )[0]

def integrate_H(l, m, n, alpha, z_nl, precomputed_N_nl):
    return sp.integrate.quadrature(
        lambda x: b(l, m, x, z_nl, precomputed_N_nl)*(-d2b(l, n, x, z_nl, precomputed_N_nl)-(2/x)*db(l, n, x, z_nl, precomputed_N_nl)+(1/x**2)*l*(l+1)*b(l, n, x, z_nl, precomputed_N_nl))*(np.log(1.0-x))**2/((1.0-x)*alpha**3), 
        0.0, 
        1.0,
        maxiter = 500
    )[0]


def initialize_physical_LE(a, rs, E_max, r0, rnn, cost_trade_off):

    l_big = 0 if rs else 50
    n_big = 50

    pad_l = 10
    pad_n = 25

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

    precomputed_N_nl = np.zeros((n_max, l_max+1))
    for n in range(n_max):
        for l in range(l_max+1):
            precomputed_N_nl[n, l] = N_nl(n, l, z_nl)

    # normalization check:
    assert np.abs(1.0-sp.integrate.quadrature(
        lambda x: get_LE_function(0, 0, x, z_nl, precomputed_N_nl)**2 * x**2, 0.0, 1.0)[0]) < 1e-6
    assert np.abs(sp.integrate.quadrature(
        lambda x: get_LE_function(0, 0, x, z_nl, precomputed_N_nl)*get_LE_function(4, 0, x, z_nl, precomputed_N_nl) * x**2, 0.0, 1.0)[0]) < 1e-6

    S = []
    for l in range(l_max+1):
        S_l = np.zeros((n_max_l[l], n_max_l[l]))

        with multiprocessing.Pool() as pool:
            results = pool.starmap(
                integrate_S,
                [(l, m, n, alpha, z_nl, precomputed_N_nl) for m in range(n_max_l[l]) for n in range(n_max_l[l])]
            )

        # fill in the matrix with the computed values
        for m in range(n_max_l[l]):
            for n in range(n_max_l[l]):
                S_l[m, n] = results[m*n_max_l[l] + n]
        # print(S_l)
        print(l)
        S.append(S_l)

    H = []
    for l in range(l_max+1):
        H_l = np.zeros((n_max_l[l], n_max_l[l]))

        with multiprocessing.Pool() as pool:
            results = pool.starmap(
                integrate_H,
                [(l, m, n, alpha, z_nl, precomputed_N_nl) for m in range(n_max_l[l]) for n in range(n_max_l[l])]
            )

        # fill in the matrix with the computed values
        for m in range(n_max_l[l]):
            for n in range(n_max_l[l]):
                H_l[m, n] = results[m*n_max_l[l] + n]
        # print(H_l)
        print(l)
        H.append(H_l)

    coeffs = []
    new_E_ln = []
    for l in range(l_max+1):
        eva, eve = sp.linalg.eigh(H[l], S[l])
        new_E_ln.append(eva)
        coeffs.append(eve)

    def function_for_splining(n, l, r):
        ret = np.zeros_like(r)
        for m in range(n_max_l[l]):
            if rnn == 0.0:
                ret += coeffs[l][m, n]*b(l, m, 1-np.exp(-r/r0), z_nl, precomputed_N_nl)
            else:
                ret += coeffs[l][m, n]*b(l, m, (1-np.exp(-r/r0))*(1.0-np.exp(-(r/rnn)**2)), z_nl, precomputed_N_nl)
        return ret

    def function_for_splining_derivative(n, l, r):
        ret = np.zeros_like(r)
        for m in range(n_max_l[l]):
            if rnn == 0.0:
                ret += coeffs[l][m, n]*db(l, m, 1-np.exp(-r/r0), z_nl, precomputed_N_nl)*np.exp(-r/r0)/r0
            else:
                ret += coeffs[l][m, n]*db(l, m, (1-np.exp(-r/r0))*(1.0-np.exp(-(r/rnn)**2)), z_nl, precomputed_N_nl)*((1.0-np.exp(-(r/rnn)**2))*np.exp(-r/r0)/r0+(1-np.exp(-r/r0))*np.exp(-(r/rnn)**2)*2*r/rnn**2)
        return ret

    if rs:
        E_nl = new_E_ln[0]
    else:
        E_nl = np.zeros((n_max, l_max+1))
        for l in range(l_max+1):
            for n in range(n_max):
                try:
                    E_nl[n, l] = new_E_ln[l][n]
                    if cost_trade_off:
                        E_nl[n, l] = E_nl[n, l]*(2*l+1)
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

    """
    import matplotlib.pyplot as plt
    r = np.linspace(0.1, a-0.001, 1000)
    plt.plot(r, function_for_splining(0, 0, r), label=str(rs))
    plt.plot([0.0, a], [0.0, 0.0], "black")
    plt.xlim(0.0, a)
    plt.legend()
    plt.savefig("radial-real.pdf")
    """

    return n_max, l_max, E_nl, spline_points
