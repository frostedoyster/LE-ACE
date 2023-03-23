import numpy as np
import torch

from equistore import TensorMap, Labels, TensorBlock


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


def process_radial_spectrum(map: TensorMap, E_n, E_max, all_species) -> TensorMap:
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
            if E_n[n] <= E_max: counter += 1
        LE_values = torch.zeros((block.values.shape[0], counter))
        if do_gradients: LE_gradients = torch.zeros((block.gradient("positions").data.shape[0], block.gradient("positions").data.shape[1], counter))
        counter_LE = 0
        counter_total = 0
        labels_LE = [] 
        for n in block.properties["n"]:
            if E_n[n] <= E_max: 
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
