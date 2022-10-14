import string
from typing import Counter
import numpy as np
import torch

from equistore import TensorMap, Labels, TensorBlock
from rascaline import SphericalExpansion


def cut_to_LE(map: TensorMap, E_nl, E_max) -> TensorMap:
    LE_blocks = []
    for idx, block in map:
        l = idx[0]
        counter = 0
        for n in block.properties["n"]:
            if E_nl[n, l] <= E_max: counter += 1
        LE_values = torch.zeros((block.values.shape[0], block.values.shape[1], counter))
        counter_LE = 0
        counter_total = 0
        labels_LE = [] 
        for n in block.properties["n"]:
            if E_nl[n, l] <= E_max: 
                LE_values[:, :, counter_LE] = torch.tensor(block.values[:, :, counter_total])
                labels_LE.append([block.properties["species_neighbor"][counter_total], n, l])
                counter_LE += 1
            counter_total += 1
        LE_block = TensorBlock(
            values=LE_values,
            samples=block.samples,
            components=block.components,
            properties=Labels(
                names = ("a1", "n1", "l1"),
                values = np.array(labels_LE),
            ),
        )
        LE_blocks.append(LE_block)
    return TensorMap(
            keys = Labels(
                names = ("lam", "a_i"),
                values = map.keys.asarray(),
            ),
            blocks = LE_blocks
        )


def get_LE_expansion(structures, spline_file: string, E_nl, E_max, rcut) -> TensorMap:

    n_max = np.where(E_nl[:, 0] <= E_max)[0][-1] + 1
    l_max = np.where(E_nl[0, :] <= E_max)[0][-1]

    hypers_spherical_expansion = {
            "cutoff": rcut,
            "max_radial": int(n_max),
            "max_angular": int(l_max),
            "center_atom_weight": 0.0,
            "radial_basis": {"Tabulated": {"file": spline_file}},
            "atomic_gaussian_width": 100.0,
            "cutoff_function": {"Step": {}},
        }

    calculator = SphericalExpansion(**hypers_spherical_expansion)
    spherical_expansion_coefficients = calculator.compute(structures)

    all_species = np.unique(spherical_expansion_coefficients.keys["species_center"])
    all_neighbor_species = Labels(
            names=["species_neighbor"],
            values=np.array(all_species, dtype=np.int32).reshape(-1, 1),
        )
    spherical_expansion_coefficients.keys_to_properties(all_neighbor_species)

    n_max_l = []
    for l in range(l_max+1):
        n_max_l.append(np.where(E_nl[:, l] <= E_max)[0][-1]+1) 

    LE_spherical = cut_to_LE(spherical_expansion_coefficients, E_nl, E_max)

    return LE_spherical


def get_LE_ps(structures, spline_file: string, E_nl, E_max_2, rcut) -> TensorMap:

    E_max_1 = E_max_2 - E_nl[0, 0]
    spherical_expansion = get_LE_expansion(structures, spline_file, E_nl, E_max_1, rcut)
    all_species = np.unique(np.concatenate([spherical_expansion.keys["a_i"], spherical_expansion.keys["a_i"]]))  # This may actually need to come from outside

    l_max = 0
    for idx, block in spherical_expansion:
        l_max = max(l_max, idx[0])

    n_max_l = []
    a_max = 0
    a_i = all_species[0]
    for l in range(l_max+1):
        old_block = spherical_expansion.block(lam=l, a_i=a_i)
        a = old_block.properties["a1"]
        n = old_block.properties["n1"]
        a_max = np.max(a) + 1
        n_max_l.append(np.max(n)+1)

    combined_anl = {}
    anl_counter = 0
    for a in range(a_max):
        for l in range(l_max+1):
            for n in range(n_max_l[l]):
                combined_anl[(a, n, l,)] = anl_counter
                anl_counter += 1

    blocks = []
    for a_i in all_species:

        soap_count = 0
        for l in range(l_max+1):
            old_block = spherical_expansion.block(lam=l, a_i=a_i)
            a = old_block.properties["a1"]
            n = old_block.properties["n1"]

            for i in range(old_block.values.shape[-1]):
                for j in range(old_block.values.shape[-1]):
                    if combined_anl[(a[i], n[i], l)] > combined_anl[(a[j], n[j], l)]: continue  # Lexicographic
                    if E_nl[n[i], l] + E_nl[n[j], l] > E_max_2: continue  # LE eigenvalue
                    soap_count += 1

        data = torch.empty((len(old_block.samples), soap_count), device=old_block.values.device)

        soap_count = 0  # reset counter
        properties_names = (
            [f"{name[:-1]}1" for name in old_block.properties.names]
            + [f"{name[:-1]}2" for name in old_block.properties.names]
        )
        properties_values = []
        for l in range(l_max + 1):  # loops over l to ensure consistent order, independent on key storage
            old_block = spherical_expansion.block(lam=l, a_i=a_i)
            a = old_block.properties["a1"]
            n = old_block.properties["n1"]

            soap_prefactor = 1.0 / np.sqrt(2 * l + 1)

            for i in range(old_block.values.shape[-1]):
                for j in range(old_block.values.shape[-1]):
                    if combined_anl[(a[i], n[i], l)] > combined_anl[(a[j], n[j], l)]: continue  # Lexicographic
                    if E_nl[n[i], l] + E_nl[n[j], l] > E_max_2: continue  # LE eigenvalue
                    multiplicity_factor = np.sqrt(2.0)
                    if combined_anl[(a[i], n[i], l)] == combined_anl[(a[j], n[j], l)]: multiplicity_factor = 1.0

                    properties_values.append([a[i], n[i], l, a[j], n[j], l])

                    data[:, soap_count] = multiplicity_factor*soap_prefactor*torch.sum(old_block.values[:, :, i]*old_block.values[:, :, j], dim = 1, keepdim = False)

                    soap_count += 1

        block = TensorBlock(
            values=data,
            samples=old_block.samples,
            components=[],
            properties=Labels(
                names=properties_names,
                values=np.asarray(np.vstack(properties_values), dtype=np.int32),
            ),
        )
        blocks.append(block)

    LE_ps = TensorMap(
        keys = Labels(
            names = ("a_i",),
            values = np.array(all_species).reshape((-1, 1)),
        ), 
        blocks = blocks)

    return LE_ps

