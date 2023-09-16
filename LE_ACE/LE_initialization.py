import numpy as np

import rascaline
import rascaline.torch
from metatensor.torch import Labels

import scipy as sp
from scipy import optimize
from scipy.special import spherical_jn as j_l
from scipy.special import spherical_in as i_l
from scipy.special import spherical_yn as y_l
from .spherical_bessel_zeros import Jn_zeros, get_laplacian_eigenvalues
from scipy.integrate import quadrature

from .physical_LE import initialize_physical_LE
from .LE_cutoff import get_LE_cutoff


def initialize_basis(a, rs, E_max, le_type, r0, rnn, cost_trade_off=False):

    # Will return eigenvalues and a calculator
    if le_type == "pure" or le_type == "paper" or le_type == "transform":
        n_max, l_max, E_nl, splines = initialize_LE(a, rs, E_max, r0, rnn, le_type, cost_trade_off)
    elif le_type == "physical":
        n_max, l_max, E_nl, splines = initialize_physical_LE(a, rs, E_max, r0, rnn, cost_trade_off)
    else:
        raise NotImplementedError("LE type can only be pure, paper, transform, and physical")

    calculator = get_calculator(a, n_max, l_max, le_type, splines)

    return l_max, E_nl, calculator


def initialize_LE(a, rs, E_max, r0, rnn, le_type, cost_trade_off=False):

    l_big = 0 if rs else 50
    n_big = 50

    E_nl = get_laplacian_eigenvalues(n_big, l_big, cost_trade_off=cost_trade_off)
    if rs:
        E_nl = E_nl[:, 0]
    n_max, l_max = get_LE_cutoff(E_nl, E_max, rs)

    z_ln = Jn_zeros(l_max, n_max)  # Spherical Bessel zeros
    z_nl = z_ln.T

    def R_nl(n, l, r):
        return j_l(l, z_nl[n, l]*r/a)

    def N_nl(n, l):
        # Normalization factor for LE basis functions
        def function_to_integrate_to_get_normalization_factor(x):
            return j_l(l, x)**2 * x**2
        integral, _ = sp.integrate.quadrature(function_to_integrate_to_get_normalization_factor, 0.0, z_nl[n, l], maxiter=100)
        return ((a/z_nl[n, l])**3 * integral)**(-0.5)

    precomputed_N_nl = np.zeros((n_max, l_max+1))
    for n in range(n_max):
        for l in range(l_max+1):
            precomputed_N_nl[n, l] = N_nl(n, l)

    def get_LE_function(n, l, r):
        R = np.zeros_like(r)
        for i in range(r.shape[0]):
            R[i] = R_nl(n, l, r[i])
        return precomputed_N_nl[n, l]*R
        '''
        # second kind
        ret = y_l(l, z_nl[n, l]*r/a)
        for i in range(len(ret)):
            if ret[i] < -1000000000.0: ret[i] = -1000000000.0

        return ret
        '''

    def radial_transform(r):
        # Function that defines the radial transform x = xi(r).
        if le_type == "pure":
            x = r
        elif le_type == "paper":
            x = a*(1.0-np.exp(-r0*np.tan(np.pi*r/(2*a))))
        elif le_type == "transform":
            if rnn == 0.0:
                x = a*(1.0-np.exp(-r/r0))
            else:
                x = a*(1.0-np.exp(-r/r0))*(1.0-np.exp(-(r/rnn)**2))
        else:
            raise NotImplementedError("LE type here can only be pure, paper, and transform")
        return x

    def get_LE_radial_transform(n, l, r):
        # Calculates radially transformed LE radial basis function for a 1D array of values r.
        x = radial_transform(r)
        return get_LE_function(n, l, x)

    # Feed LE (delta) radial spline points to Rust calculator:

    def function_for_splining(n, l, r):
        return get_LE_radial_transform(n, l, r)

    def function_for_splining_derivative(n, l, r):
        delta = 1e-6
        all_derivatives_except_first_and_last = (function_for_splining(n, l, r[1:-1]+delta) - function_for_splining(n, l, r[1:-1]-delta)) / (2.0*delta)
        derivative_at_zero = (function_for_splining(n, l, np.array([delta/10.0])) - function_for_splining(n, l, np.array([0.0]))) / (delta/10.0)
        derivative_last = (function_for_splining(n, l, np.array([a])) - function_for_splining(n, l, np.array([a-delta/10.0]))) / (delta/10.0)
        return np.concatenate([derivative_at_zero, all_derivatives_except_first_and_last, derivative_last])

    spline_points = rascaline.generate_splines(
        function_for_splining,
        function_for_splining_derivative,
        n_max,
        l_max,
        a,
        requested_accuracy = 1e-8
    )
    print("Number of spline points:", len(spline_points))

    return n_max, l_max, E_nl, spline_points


def get_calculator(a, n_max, l_max, le_type, spline_points):

    hypers_spherical_expansion = {
            "cutoff": a,
            "max_radial": int(n_max),
            "max_angular": int(l_max),
            "center_atom_weight": 0.0,
            "radial_basis": {"TabulatedRadialIntegral": {"points": spline_points}},
            "atomic_gaussian_width": 100.0,
        }
    
    if le_type == "pure":
        hypers_spherical_expansion["cutoff_function"] = {"ShiftedCosine": {"width": 1.0}}
    elif le_type == "paper":
        hypers_spherical_expansion["cutoff_function"] = {"Step": {}}
    elif le_type == "transform":
        hypers_spherical_expansion["cutoff_function"] = {"ShiftedCosine": {"width": 1.0}}
    elif le_type == "physical":
        hypers_spherical_expansion["cutoff_function"] = {"ShiftedCosine": {"width": 1.0}}
    else:
        raise NotImplementedError("LE types can only be pure, paper, and radial_transform")

    if l_max == 0:
        hypers_spherical_expansion.pop("max_angular")
        calculator = rascaline.torch.SoapRadialSpectrum(**hypers_spherical_expansion)
    else:
        calculator = rascaline.torch.SphericalExpansion(**hypers_spherical_expansion)


    # Uncomment this to inspect the spherical expansion
    """
    if l_max != 0:
        import ase

        def get_dummy_structures(r_array):
            dummy_structures = []
            for r in r_array:
                dummy_structures.append(
                    ase.Atoms('CH', positions=[(0, 0, 0), (0, 0, r)])
                )
            return dummy_structures 

        # Create a fake list of dummy structures to test the radial functions generated from rascaline.torch.

        r = np.linspace(0.1, a-0.001, 1000)
        structures = get_dummy_structures(r)

        spherical_expansion_coefficients = calculator.compute(structures)

        block_C_0 = spherical_expansion_coefficients.block(species_center = 6, spherical_harmonics_l = 0, species_neighbor = 1)
        # print(block_C_0.values.shape)

        block_C_0_0 = block_C_0.values[:, :, 0].flatten()
        spherical_harmonics_0 = 1.0/np.sqrt(4.0*np.pi)

        all_species = np.unique(spherical_expansion_coefficients.keys["species_center"])
        all_neighbor_species = Labels(
                names=["species_neighbor"],
                values=np.array(all_species, dtype=np.int32).reshape(-1, 1),
            )
        spherical_expansion_coefficients = spherical_expansion_coefficients.keys_to_properties(all_neighbor_species)

        import matplotlib.pyplot as plt
        plt.plot(r, block_C_0_0/spherical_harmonics_0, label="rascaline.torch output")
        plt.plot([0.0, a], [0.0, 0.0], "black")
        plt.xlim(0.0, a)
        plt.legend()
        plt.savefig("radial.pdf")
        """

    return calculator
