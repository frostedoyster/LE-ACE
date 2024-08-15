import numpy as np
import matplotlib.pyplot as plt
import os

import rascaline

from ..LE_cutoff import get_LE_cutoff


# All these periodic functions are zeroed for the (unlikely) case where r > 10*r_0
# which is outside the domain where the eigenvalue equation was solved

def s(n, r, a):
    return np.where(
        r < a,
        np.sin(np.pi*(n+1.0)*r/a),
        0.0
    )

def ds(n, r, a):
    return np.where(
        r < a,
        (np.pi*(n+1.0)/a)*np.cos(np.pi*(n+1.0)*r/a),
        0.0
    )

def c(n, r, a):
    return np.where(
        r < a,
        np.cos(np.pi*(n+0.5)*r/a),
        0.0
    )

def dc(n, r, a):
    return np.where(
        r < a,
        -(np.pi*(n+0.5)/a)*np.sin(np.pi*(n+0.5)*r/a),
        0.0
    )


def initialize_physical_LE(r_cut, rs, E_max, r_0, rnn, cost_trade_off):

    l_max = 50
    n_max = 50
    n_max_big = 200

    a = 10.0*r_0  # the solutions we load are scaled by r_0

    dir_path = os.path.dirname(os.path.realpath(__file__))

    E_ln = np.load(
        os.path.join(
            dir_path,
            "eigenvalues.npy"
        )
    )
    eigenvectors = np.load(        
        os.path.join(
            dir_path,
            "eigenvectors.npy"
        )
    )

    E_nl = E_ln.T
    if rs:
        E_n0 = E_nl[:, 0]
    n_max, l_max = get_LE_cutoff(E_nl, E_max, rs)

    def function_for_splining(n, l, r):
        ret = np.zeros_like(r)
        for m in range(n_max_big):
            ret += (eigenvectors[l][m, n]*c(m, r, a) if l == 0 else eigenvectors[l][m, n]*s(m, r, a))
        return ret

    def function_for_splining_derivative(n, l, r):
        ret = np.zeros_like(r)
        for m in range(n_max_big):
            ret += (eigenvectors[l][m, n]*dc(m, r, a) if l == 0 else eigenvectors[l][m, n]*ds(m, r, a))
        return ret

    spliner = rascaline.utils.RadialIntegralFromFunction(
        radial_integral=function_for_splining,
        radial_integral_derivative=function_for_splining_derivative,
        max_radial=n_max,
        max_angular=l_max,
        spline_cutoff=a,
        accuracy=1e-6,
    )
    spline_points = spliner.compute()

    """
    import matplotlib.pyplot as plt
    r = np.linspace(0.1, r_cut-0.001, 1000)
    plt.plot(r, function_for_splining(0, 0, r), label=str(rs))
    plt.plot([0.0, r_cut], [0.0, 0.0], "black")
    plt.xlim(0.0, r_cut)
    plt.legend()
    plt.savefig("radial-real.pdf")
    """

    if rs: E_nl = E_ln[0]  # E_n0 as a 1D array

    return n_max, l_max, E_nl, spline_points
