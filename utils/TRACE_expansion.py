import numpy as np
import torch

from metatensor.torch import TensorMap, Labels, TensorBlock


def process_spherical_expansion(map: TensorMap, E_nl, E_max, all_species, device) -> TensorMap:
    do_gradients = map.block(0).has_gradient("positions")

    species_remapping = {}
    for i_species, species in enumerate(all_species):
        species_remapping[species] = i_species

    TRACE_blocks = []
    for idx, block in map.items():

        l = idx[0]
        counter = 0
        for n in block.properties["n"]:
            if E_nl[n, l] <= E_max: counter += 1
        TRACE_values = torch.zeros((block.values.shape[0], block.values.shape[1], counter))
        if do_gradients: TRACE_gradients = torch.zeros((block.gradient("positions").values.shape[0], block.gradient("positions").values.shape[1], block.values.shape[1], counter))
        counter_TRACE = 0
        counter_total = 0
        labels_TRACE = [] 
        for n in block.properties["n"]:
            if E_nl[n, l] <= E_max: 
                TRACE_values[:, :, counter_TRACE] = torch.tensor(block.values[:, :, counter_total])
                if do_gradients: TRACE_gradients[:, :, :, counter_TRACE] = torch.tensor(block.gradient("positions").values[:, :, :, counter_total])
                labels_TRACE.append([species_remapping[block.properties["species_neighbor"][counter_total]], n, l, l])
                counter_TRACE += 1
            counter_total += 1
        TRACE_block = TensorBlock(
            values=TRACE_values.to(device),
            samples=block.samples,
            components=block.components,
            properties=Labels(
                names = ("a1", "n1", "l1", "k1"),
                values = np.array(labels_TRACE),
            ),
        )
        if do_gradients: TRACE_block.add_gradient(
            parameter="positions",
            gradient=TensorBlock(
                values = TRACE_gradients.to(device), 
                samples = block.gradient("positions").samples, 
                components = block.gradient("positions").components,
                properties=TRACE_block.properties
            )
        )
        TRACE_blocks.append(TRACE_block)
    return TensorMap(
            keys = Labels(
                names = ("lam", "a_i"),
                values = map.keys.values,
            ),
            blocks = TRACE_blocks
        )


def process_radial_spectrum(map: TensorMap, E_n, E_max, all_species) -> TensorMap:
    # TODO: This could really be simplified: no TRACE-like cutting should be necessary

    do_gradients = map.block(0).has_gradient("positions")

    species_remapping = {}
    for i_species, species in enumerate(all_species):
        species_remapping[species] = i_species

    TRACE_blocks = []
    for species_center in all_species:  # TODO: do not rely on all_species, which is used for the neighbors but some may be missing from the centers
        block = map.block(species_center=species_center)
        l = 0
        counter = 0
        for n in block.properties["n"]:
            if E_n[n] <= E_max: counter += 1
        TRACE_values = torch.zeros((block.values.shape[0], counter))
        if do_gradients: TRACE_gradients = torch.zeros((block.gradient("positions").values.shape[0], block.gradient("positions").values.shape[1], counter))
        counter_TRACE = 0
        counter_total = 0
        labels_TRACE = [] 
        for n in block.properties["n"]:
            if E_n[n] <= E_max: 
                TRACE_values[:, counter_TRACE] = torch.tensor(block.values[:, counter_total])
                if do_gradients: TRACE_gradients[:, :, counter_TRACE] = torch.tensor(block.gradient("positions").values[:, :, counter_total])
                labels_TRACE.append([species_remapping[block.properties["species_neighbor"][counter_total]], n, l, l])
                counter_TRACE += 1
            counter_total += 1
        TRACE_block = TensorBlock(
            values=TRACE_values,
            samples=block.samples,
            components=block.components,
            properties=Labels(
                names = ("a1", "n1", "l1", "k1"),
                values = np.array(labels_TRACE),
            ),
        )
        if do_gradients: TRACE_block.add_gradient(
            parameter="positions",
            gradient=TensorBlock(
                values = TRACE_gradients, 
                samples = block.gradient("positions").samples, 
                components = block.gradient("positions").components,
                properties = TRACE_block.properties,
            )
        )
        TRACE_blocks.append(TRACE_block)

    return TensorMap(
            keys = Labels(
                names = ("a_i",),
                values = map.keys.values,
            ),
            blocks = TRACE_blocks
        )


def get_TRACE_expansion(structures, calculator, E_nl, E_max, all_species, contraction_matrix, rs=False, do_gradients=False, device="cpu") -> TensorMap:

    n_trace = contraction_matrix.shape[1]

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

    n_elems = len(all_species)

    new_blocks = []
    for key, block in spherical_expansion_coefficients.items():

        new_values = (
            block.values.reshape(
                block.values.shape[:-1] + (n_elems, -1)
        ).swapaxes(-2, -1) @ contraction_matrix).swapaxes(-2, -1).reshape(block.values.shape[:-1] +(-1,))

        new_properties_values = np.array([
            [z, block.properties[nl_index][1], block.properties[nl_index][2], block.properties[nl_index][3]] for z in range(n_trace) for nl_index in range(len(block.properties)//n_elems)
        ], dtype = np.int32)

        new_properties = Labels(
            names = ["z", "n1", "l1", "k1"],
            values = new_properties_values,
        )

        new_block = TensorBlock(
            values = new_values,
            samples = block.samples,
            components = block.components,
            properties = new_properties
        )

        if do_gradients:
            new_gradients = (block.gradient("positions").values.reshape(
                block.gradient("positions").values.shape[:-1] + (n_elems, -1)
            ).swapaxes(-2, -1) @ contraction_matrix).swapaxes(-2, -1).reshape(block.gradient("positions").values.shape[:-1] + (-1,))

            new_block.add_gradient(
                parameter="positions",
                gradient=TensorBlock(
                    values = new_gradients,
                    samples = block.gradient("positions").samples, 
                    components = block.gradient("positions").components,
                    properties = new_properties
                )
            )

        new_blocks.append(new_block)

    spherical_expansion_coefficients = TensorMap(
        keys = spherical_expansion_coefficients.keys,
        blocks = new_blocks
    )
        
    return spherical_expansion_coefficients
