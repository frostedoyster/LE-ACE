import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.special import spherical_jn as j_l
from scipy.special import spherical_in as i_l
from scipy.integrate import quadrature

from .spherical_bessel_zeros import Jn_zeros


def get_coefficients(l_max, n_max_required, r0):

    n_max = n_max_required + 10
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
        S_l = np.zeros((n_max, n_max))
        for m in range(n_max):
            for n in range(n_max):
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
        H_l = np.zeros((n_max, n_max))
        for m in range(n_max):
            for n in range(n_max):
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

    c = np.zeros((l_max+1, n_max, n_max))
    for l in range(l_max+1):
        eva, eve = sp.linalg.eigh(H[l], S[l])
        for n in range(n_max):
            for m in range(n_max):
                c[l, n, m] = eve[:, n][m]

    return c
