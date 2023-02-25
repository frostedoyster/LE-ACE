import numpy as np
import torch

from equistore import TensorMap, Labels, TensorBlock
import rascaline

import scipy as sp
from scipy import optimize
from scipy.special import spherical_jn as j_l
from scipy.special import spherical_in as i_l
from scipy.special import spherical_yn as y_l
from .spherical_bessel_zeros import Jn_zeros
from scipy.integrate import quadrature

import os 
import json


def process_spherical_expansion(map: TensorMap, E_nl, E_max, all_species, device) -> TensorMap:
    do_gradients = map.block(0).has_gradient("positions")

    species_remapping = {}
    for i_species, species in enumerate(all_species):
        species_remapping[species] = i_species

    LE_blocks = []
    for idx, block in map:

        l = idx[0]
        counter = 0
        for n in block.properties["n"]:
            if E_nl[n, l] <= E_max: counter += 1
        LE_values = torch.zeros((block.values.shape[0], block.values.shape[1], counter))
        if do_gradients: LE_gradients = torch.zeros((block.gradient("positions").data.shape[0], block.gradient("positions").data.shape[1], block.values.shape[1], counter))
        counter_LE = 0
        counter_total = 0
        labels_LE = [] 
        for n in block.properties["n"]:
            if E_nl[n, l] <= E_max: 
                LE_values[:, :, counter_LE] = torch.tensor(block.values[:, :, counter_total])
                if do_gradients: LE_gradients[:, :, :, counter_LE] = torch.tensor(block.gradient("positions").data[:, :, :, counter_total])
                labels_LE.append([species_remapping[block.properties["species_neighbor"][counter_total]], n, l, l])
                counter_LE += 1
            counter_total += 1
        LE_block = TensorBlock(
            values=LE_values.to(device),
            samples=block.samples,
            components=block.components,
            properties=Labels(
                names = ("a1", "n1", "l1", "k1"),
                values = np.array(labels_LE),
            ),
        )
        if do_gradients: LE_block.add_gradient(
            "positions",
            data = LE_gradients.to(device), 
            samples = block.gradient("positions").samples, 
            components = block.gradient("positions").components,
        )
        LE_blocks.append(LE_block)
    return TensorMap(
            keys = Labels(
                names = ("lam", "a_i"),
                values = map.keys.asarray(),
            ),
            blocks = LE_blocks
        )


def process_radial_spectrum(map: TensorMap, E_nl, E_max, all_species) -> TensorMap:
    # TODO: This could really be simplified: no LE-like cutting should be necessary

    do_gradients = map.block(0).has_gradient("positions")

    species_remapping = {}
    for i_species, species in enumerate(all_species):
        species_remapping[species] = i_species

    LE_blocks = []
    for species_center in all_species:  # TODO: do not rely on all_species, which is used for the neighbors but some may be missing from the centers
        block = map.block(species_center=species_center)
        l = 0
        counter = 0
        for n in block.properties["n"]:
            if E_nl[n, l] <= E_max: counter += 1
        LE_values = torch.zeros((block.values.shape[0], counter))
        if do_gradients: LE_gradients = torch.zeros((block.gradient("positions").data.shape[0], block.gradient("positions").data.shape[1], counter))
        counter_LE = 0
        counter_total = 0
        labels_LE = [] 
        for n in block.properties["n"]:
            if E_nl[n, l] <= E_max: 
                LE_values[:, counter_LE] = torch.tensor(block.values[:, counter_total])
                if do_gradients: LE_gradients[:, :, counter_LE] = torch.tensor(block.gradient("positions").data[:, :, counter_total])
                labels_LE.append([species_remapping[block.properties["species_neighbor"][counter_total]], n, l, l])
                counter_LE += 1
            counter_total += 1
        LE_block = TensorBlock(
            values=LE_values,
            samples=block.samples,
            components=block.components,
            properties=Labels(
                names = ("a1", "n1", "l1", "k1"),
                values = np.array(labels_LE),
            ),
        )
        if do_gradients: LE_block.add_gradient(
            "positions",
            data = LE_gradients, 
            samples = block.gradient("positions").samples, 
            components = block.gradient("positions").components,
        )
        LE_blocks.append(LE_block)

    return TensorMap(
            keys = Labels(
                names = ("a_i",),
                values = map.keys.asarray(),
            ),
            blocks = LE_blocks
        )


def get_LE_expansion(structures, calculator, E_nl, E_max, all_species, rs=False, do_gradients=False, device="cpu") -> TensorMap:

    gradients = (["positions"] if do_gradients else None)
    spherical_expansion_coefficients = calculator.compute(structures, gradients=gradients)

    all_neighbor_species = Labels(
            names=["species_neighbor"],
            values=np.array(all_species, dtype=np.int32).reshape(-1, 1),
        )
    spherical_expansion_coefficients = spherical_expansion_coefficients.keys_to_properties(all_neighbor_species)

    if rs:
        spherical_expansion_coefficients = process_radial_spectrum(spherical_expansion_coefficients, E_nl, E_max, all_species)
    else:
        spherical_expansion_coefficients = process_spherical_expansion(spherical_expansion_coefficients, E_nl, E_max, all_species, device=device)
        
    return spherical_expansion_coefficients


def get_calculator(a, n_max, l_max, factor):

    l_big = 50
    n_big = 50

    z_ln = Jn_zeros(l_big, n_big)  # Spherical Bessel zeros
    z_nl = z_ln.T

    def R_nl(n, l, r):
        return j_l(l, z_nl[n, l]*r/a)

    def N_nl(n, l):
        # Normalization factor for LE basis functions
        def function_to_integrate_to_get_normalization_factor(x):
            return j_l(l, x)**2 * x**2
        # print(f"Trying to integrate n={n} l={l}")
        integral, _ = sp.integrate.quadrature(function_to_integrate_to_get_normalization_factor, 0.0, z_nl[n, l], maxiter=100)
        return (1.0/z_nl[n, l]**3 * integral)**(-0.5)

    def get_LE_function(n, l, r):
        R = np.zeros_like(r)
        for i in range(r.shape[0]):
            R[i] = R_nl(n, l, r[i])
        return N_nl(n, l)*R*a**(-1.5)  # This is what makes the results different when you increasing a indefinitely.
        '''
        # second kind
        ret = y_l(l, z_nl[n, l]*r/a)
        for i in range(len(ret)):
            if ret[i] < -1000000000.0: ret[i] = -1000000000.0

        return ret
        '''

    # Radial transform
    def radial_transform(r):
        # Function that defines the radial transform x = xi(r).
        x = a*(1.0-np.exp(-factor*np.tan(np.pi*r/(2*a))))
        # x = a*(1.0-np.exp(-r/factor))*(1.0-np.exp(-(r/factor2)**2))
        return x

    def get_LE_radial_transform(n, l, r):
        # Calculates radially transformed LE radial basis function for a 1D array of values r.
        x = radial_transform(r)
        return get_LE_function(n, l, x) # *np.exp(-r/factor)

    def cutoff_function(r):
        cutoff = 3.0
        width = 0.5
        ret = np.zeros_like(r)
        for i, single_r in enumerate(r):
            ret[i] = (0.5*(1.0+np.cos(np.pi*(single_r-cutoff+width)/width)) if single_r > cutoff-width else 1.0)
        return ret

    def radial_scaling(r):
        rate = 1.0
        scale = 2.0
        exponent = 7.0
        return rate / (rate + (r / scale) ** exponent)

    def get_LE_radial_scaling(n, l, r):
        return get_LE_function(n, l, r)*radial_scaling(r)*cutoff_function(r)

    # Feed LE (delta) radial spline points to Rust calculator:

    def function_for_splining(n, l, x):
        return get_LE_radial_transform(n, l, x)
        # return get_LE_function(n, l, x)

    def function_for_splining_derivative(n, l, r):
        delta = 1e-6
        all_derivatives_except_first_and_last = (function_for_splining(n, l, r[1:-1]+delta) - function_for_splining(n, l, r[1:-1]-delta)) / (2.0*delta)
        derivative_at_zero = (function_for_splining(n, l, np.array([delta/10.0])) - function_for_splining(n, l, np.array([0.0]))) / (delta/10.0)
        derivative_last = (function_for_splining(n, l, np.array([a])) - function_for_splining(n, l, np.array([a-delta/10.0]))) / (delta/10.0)
        return np.concatenate([derivative_at_zero, all_derivatives_except_first_and_last, derivative_last])
    

    spline_path = f"splines/splines-{l_max}-{n_max}-{a}-{factor}.json"
    if os.path.exists(spline_path):
        with open(spline_path, 'r') as file:
            spline_points = json.load(file)
    else:
        spline_points = rascaline.generate_splines(
            function_for_splining,
            function_for_splining_derivative,
            n_max,
            l_max,
            a,
            requested_accuracy = 1e-6
        )
        with open(spline_path, 'w') as file:
            json.dump(spline_points, file)

    print("Number of spline points:", len(spline_points))

    hypers_spherical_expansion = {
            "cutoff": a,
            "max_radial": int(n_max),
            "max_angular": int(l_max),
            "center_atom_weight": 0.0,
            "radial_basis": {"TabulatedRadialIntegral": {"points": spline_points}},
            "atomic_gaussian_width": 100.0,
            "cutoff_function": {
                "Step": {},
                #"ShiftedCosine": {"width": 0.5},
            },
        }

    if l_max == 0:
        hypers_spherical_expansion.pop("max_angular")
        calculator = rascaline.SoapRadialSpectrum(**hypers_spherical_expansion)
    else:
        calculator = rascaline.SphericalExpansion(**hypers_spherical_expansion)


    # Uncomment this to inspect the spherical expanaion
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

        # Create a fake list of dummy structures to test the radial functions generated from rascaline.

        r = np.linspace(0.1, a-0.001, 1000)
        structures = get_dummy_structures(r)

        spherical_expansion_coefficients = calculator.compute(structures)

        block_C_0 = spherical_expansion_coefficients.block(species_center = 6, spherical_harmonics_l = 0, species_neighbor = 1)
        print(block_C_0.values.shape)

        block_C_0_0 = block_C_0.values[:, :, 4].flatten()
        spherical_harmonics_0 = 1.0/np.sqrt(4.0*np.pi)

        all_species = np.unique(spherical_expansion_coefficients.keys["species_center"])
        all_neighbor_species = Labels(
                names=["species_neighbor"],
                values=np.array(all_species, dtype=np.int32).reshape(-1, 1),
            )
        spherical_expansion_coefficients.keys_to_properties(all_neighbor_species)

        import matplotlib.pyplot as plt
        plt.plot(r, block_C_0_0/spherical_harmonics_0, label="rascaline output")  # rascaline bug?
        plt.plot([0.0, a], [0.0, 0.0], "black")
        plt.plot(r, function_for_splining(n=4, l=0, x=r), "--", label="original function")
        plt.xlim(0.0, a)
        plt.legend()
        plt.savefig(f"radial.pdf")
    """

    return calculator
