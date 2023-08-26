import numpy as np
import torch
import rascaline
from .LE_initialization import initialize_basis
from .LE_metadata import get_le_metadata
from .generalized_cgs import get_generalized_cgs
from .LE_iterator import LEIterator
from .ACE_symmetrizer import ACESymmetrizer
from equistore import Labels

from .LE_expansion import get_LE_expansion
from .TRACE_expansion import get_TRACE_expansion


class LE_ACE(torch.nn.Module):

    def __init__(
        self,
        r_cut_rs,
        r_cut,
        E_max,
        all_species,
        le_type,
        factor,
        factor2,
        cost_trade_off,
        is_trace,
        device
    ):
        super().__init__()

        nu_max = len(E_max) - 1
        self.E_max = E_max
        self.nu_max = nu_max
        self.device = device
        self.is_trace = is_trace

        self.all_species = all_species
        self.n_species = len(all_species)
        self.all_species_tensor = torch.tensor(all_species, dtype=torch.long, device=self.device)
        self.species_to_species_index = torch.zeros(max(all_species)+1, dtype=torch.long, device=self.device)
        counter = 0
        for species in all_species:
            self.species_to_species_index[species] = counter
            counter +=1
        print("Self species to species index", self.species_to_species_index)

        _, self.E_n0, radial_spectrum_calculator = initialize_basis(r_cut_rs, True, E_max[1], le_type, factor, factor2)
        self.n_max_rs = np.where(self.E_n0 <= E_max[1])[0][-1] + 1
        l_max, self.E_nl, spherical_expansion_calculator = initialize_basis(r_cut, False, E_max[2], le_type, factor, factor2, cost_trade_off=cost_trade_off)

        E_ln = self.E_nl.T
        n_max_l = []
        for l in range(l_max+1):
            n_max_l.append(np.where(self.E_nl[:, l] <= E_max[2])[0][-1] + 1)
        print(self.n_max_rs)
        print(n_max_l)
        E_ln = [E_ln[l][:n_max_l[l]] for l in range(l_max+1)]

        self.combine_indices, self.multiplicities, self.LE_energies = get_le_metadata(all_species, E_max, n_max_l, E_ln, is_trace, device)

        l_tuples = [0, 1] + [list(self.combine_indices[nu].keys()) for nu in range(2, nu_max+1)]
        requested_L_sigma_pairs = [(0, 1)]
        self.generalized_cgs = get_generalized_cgs(l_tuples, requested_L_sigma_pairs, nu_max, device)
        necessary_l_tuples_nu_max = []
        for requested_L_sigma_pair in requested_L_sigma_pairs:
            necessary_l_tuples_nu_max.extend(list(self.generalized_cgs[requested_L_sigma_pair][nu_max].keys()))

        # For correlation order nu, remove unncecessary indices that violate the parity constraints:
        keys_to_pop = []
        for l_tuple_nu_max in self.combine_indices[nu_max].keys():
            if l_tuple_nu_max not in necessary_l_tuples_nu_max:
                keys_to_pop.append(l_tuple_nu_max)
        
        for l_tuple_nu_max in keys_to_pop:
            self.combine_indices[nu_max].pop(l_tuple_nu_max)
            self.multiplicities[nu_max].pop(l_tuple_nu_max)
            self.LE_energies[nu_max].pop(l_tuple_nu_max)

        self.fixed_order_l_tuples = [0, 1] + [list(self.generalized_cgs[(0, 1)][nu].keys()) for nu in range(2, nu_max+1)]  # Needed to have consistent ordering when concatenating

        # Build extended LE_energies according to L_tuple and a_i:
        self.extended_LE_energies = []
        for nu in range(self.nu_max+1):
            if nu == 0:
                extended_LE_energies_nu = torch.tensor([0.0], dtype=torch.get_default_dtype(), device=self.device)
            elif nu == 1:
                extended_LE_energies_nu = torch.tile(
                    torch.tensor(self.E_n0[:self.n_max_rs], dtype=torch.get_default_dtype(), device=self.device),
                    (self.n_species**2,)
                )
            else:
                extended_LE_energies_nu = []
                for l_tuple in self.fixed_order_l_tuples[nu]:
                    extended_LE_energies_nu.append(self.LE_energies[nu][l_tuple])
                extended_LE_energies_nu = torch.concatenate(extended_LE_energies_nu)
                extended_LE_energies_nu = torch.tile(
                    extended_LE_energies_nu,
                    (self.n_species*self.generalized_cgs[(0, 1)][nu][l_tuple].shape[0],)  # Account for L (as well as a_i)
                )
            self.extended_LE_energies.append(extended_LE_energies_nu)
        print([tensor.shape[0] for tensor in self.extended_LE_energies])

        self.nu0_calculator_train = rascaline.AtomicComposition(per_structure=False)
        self.radial_spectrum_calculator_train = radial_spectrum_calculator
        self.spherical_expansion_calculator_train = spherical_expansion_calculator
        self.le_iterator = LEIterator(self.combine_indices, self.multiplicities)
        self.ace_symmetrizer = ACESymmetrizer(self.generalized_cgs)

    def forward(self, structures):
        n_structures = len(structures)

        composition_features_tmap = self.nu0_calculator_train.compute(structures)
        composition_features_tmap = composition_features_tmap.keys_to_samples("species_center")

        if self.is_trace:
            radial_spectrum_tmap = get_TRACE_expansion(structures, radial_spectrum_calculator_train, self.E_n0, E_max[1], all_species, trace_comb, rs=True, device=self.device)
            spherical_expansion_tmap = get_TRACE_expansion(structures, spherical_expansion_calculator_train, E_nl, E_max[2], all_species, trace_comb, device=self.device)
        else:
            radial_spectrum_tmap = get_LE_expansion(structures, self.radial_spectrum_calculator_train, self.E_n0, self.E_max[1], self.all_species, rs=True, device=self.device)
            spherical_expansion_tmap = get_LE_expansion(structures, self.spherical_expansion_calculator_train, self.E_nl, self.E_max[2], self.all_species, device=self.device)

        radial_spectrum_tmap = radial_spectrum_tmap.keys_to_samples("a_i")
        spherical_expansion_tmap = spherical_expansion_tmap.keys_to_samples("a_i")

        comp_metadata = torch.tensor(composition_features_tmap.block(0).samples.values, dtype=torch.long, device=self.device)
        rs_metadata = torch.tensor(radial_spectrum_tmap.block(0).samples.values, dtype=torch.long, device=self.device)
        spex_metadata = torch.tensor(spherical_expansion_tmap.block(0).samples.values, dtype=torch.long, device=self.device)

        composition_features = torch.tensor(composition_features_tmap.block(0).values, dtype=torch.get_default_dtype(), device=self.device)
        radial_spectrum = radial_spectrum_tmap.block(0).values.to(self.device)  # NEEDED DUE TO BUG
        spherical_expansion = {(l,): block.values.swapaxes(0, 2) for (l,), block in spherical_expansion_tmap.items()}

        print("Calculating A basis")
        A_basis = self.le_iterator(spherical_expansion)
        print("Calculating B basis")
        B_basis = self.ace_symmetrizer(A_basis)[(0, 1)]
        
        # Concatenate different features at the same body_order:
        B_basis_concatenated = [composition_features, radial_spectrum]
        for nu in range(2, self.nu_max+1):
            features_nu = [B_basis[nu][l_tuple].squeeze(0) for l_tuple in self.fixed_order_l_tuples[nu]]
            B_basis_concatenated.append(
                torch.concatenate(features_nu).T
            )

        # Sum over like-atoms in each structure:

        sum_indices_comp = self.n_species*comp_metadata[:, 0]+self.species_to_species_index[comp_metadata[:, 2]]
        sum_indices_rs = self.n_species*rs_metadata[:, 0]+self.species_to_species_index[rs_metadata[:, 2]]
        sum_indices_spex = self.n_species*spex_metadata[:, 0]+self.species_to_species_index[spex_metadata[:, 2]]
        B_basis_per_structure = []
        for nu in range(0, self.nu_max+1):
            if nu == 0:
                B_basis_per_structure_nu = torch.zeros(
                    n_structures*self.n_species, B_basis_concatenated[nu].shape[1], dtype=torch.get_default_dtype(), device=self.device
                ).index_add_(0, sum_indices_comp, B_basis_concatenated[nu])
                B_basis_per_structure_nu = B_basis_per_structure_nu.reshape(n_structures, -1).sum(dim=1)
            elif nu == 1:
                B_basis_per_structure_nu = torch.zeros(
                    n_structures*self.n_species, B_basis_concatenated[nu].shape[1], dtype=torch.get_default_dtype(), device=self.device
                ).index_add_(0, sum_indices_rs, B_basis_concatenated[nu])
            else:
                B_basis_per_structure_nu = torch.zeros(
                    n_structures*self.n_species, B_basis_concatenated[nu].shape[1], dtype=torch.get_default_dtype(), device=self.device
                ).index_add_(0, sum_indices_spex, B_basis_concatenated[nu])
            B_basis_per_structure.append(
                B_basis_per_structure_nu.reshape(n_structures, -1)
            )
        
        # Concatenate different body-orders:
        print("Number of features per body-order:", [B_basis_per_structure[nu].shape[1] for nu in range(0, self.nu_max+1)])
        B_basis_all_together = torch.concatenate([B_basis_per_structure[nu] for nu in range(0, self.nu_max+1)], dim=1)

        return B_basis_all_together

    def compute_with_gradients(self, structures):

        n_structures = len(structures)
        gradients = ["positions"]

        composition_features_tmap = self.nu0_calculator_train.compute(structures, gradients=gradients)
        composition_features_tmap = composition_features_tmap.keys_to_samples("species_center")

        if self.is_trace:
            radial_spectrum_tmap = get_TRACE_expansion(structures, radial_spectrum_calculator_train, self.E_n0, E_max[1], all_species, trace_comb, rs=True, do_gradients=True)
            spherical_expansion_tmap = get_TRACE_expansion(structures, spherical_expansion_calculator_train, E_nl, E_max[2], all_species, trace_comb, do_gradients=True, device=self.device)
        else:
            radial_spectrum_tmap = get_LE_expansion(structures, self.radial_spectrum_calculator_train, self.E_n0, self.E_max[1], self.all_species, rs=True, do_gradients=True)
            spherical_expansion_tmap = get_LE_expansion(structures, self.spherical_expansion_calculator_train, self.E_nl, self.E_max[2], self.all_species, do_gradients=True, device=self.device)

        radial_spectrum_tmap = radial_spectrum_tmap.keys_to_samples("a_i")
        spherical_expansion_tmap = spherical_expansion_tmap.keys_to_samples("a_i")

        comp_metadata = torch.tensor(composition_features_tmap.block(0).samples.values, dtype=torch.long, device=self.device)
        rs_metadata = torch.tensor(radial_spectrum_tmap.block(0).samples.values, dtype=torch.long, device=self.device)
        spex_metadata = torch.tensor(spherical_expansion_tmap.block(0).samples.values, dtype=torch.long, device=self.device)

        composition_features = torch.tensor(composition_features_tmap.block(0).values, dtype=torch.get_default_dtype(), device=self.device)
        radial_spectrum = radial_spectrum_tmap.block(0).values.to(self.device)  # DUE TO BUG
        spherical_expansion = {(l,): block.values.swapaxes(0, 2) for (l,), block in spherical_expansion_tmap.items()}

        comp_grad_metadata = torch.tensor(composition_features_tmap.block(0).gradient("positions").samples.values, dtype=torch.long, device=self.device)
        rs_grad_metadata = torch.tensor(radial_spectrum_tmap.block(0).gradient("positions").samples.values, dtype=torch.long, device=self.device)
        spex_grad_metadata = torch.tensor(spherical_expansion_tmap.block(0).gradient("positions").samples.values, dtype=torch.long, device=self.device)

        composition_features_grad = torch.tensor(composition_features_tmap.block(0).gradient("positions").values, dtype=torch.get_default_dtype(), device=self.device)
        radial_spectrum_grad = radial_spectrum_tmap.block(0).gradient("positions").values.to(self.device)  # DUE TO BUG
        spherical_expansion_grad = {
            (l,): block.gradient("positions").values.reshape(-1, block.gradient("positions").values.shape[2], block.gradient("positions").values.shape[3]).swapaxes(0, 2).reshape(
                block.gradient("positions").values.shape[3],
                block.gradient("positions").values.shape[2],
                block.gradient("positions").values.shape[0],
                block.gradient("positions").values.shape[1]
            )
            for (l,), block in spherical_expansion_tmap.items()
        }

        print("Calculating A basis")
        A_basis, A_basis_gradients = self.le_iterator.compute_with_gradients(spherical_expansion, spherical_expansion_grad, spex_grad_metadata[:, 0])
        print("Calculating B basis")
        B_basis, B_basis_gradients = self.ace_symmetrizer.compute_with_gradients(A_basis, A_basis_gradients)
        B_basis = B_basis[(0, 1)]
        B_basis_grad = B_basis_gradients[(0, 1)]
        
        # Concatenate different features at the same body_order:
        B_basis_concatenated = [composition_features, radial_spectrum]
        for nu in range(2, self.nu_max+1):
            features_nu = [B_basis[nu][l_tuple].squeeze(0) for l_tuple in self.fixed_order_l_tuples[nu]]
            B_basis_concatenated.append(
                torch.concatenate(features_nu).T
            )
        B_basis_concatenated_grad = [composition_features_grad, radial_spectrum_grad]
        for nu in range(2, self.nu_max+1):
            gradients_nu = [B_basis_grad[nu][l_tuple].squeeze(0) for l_tuple in self.fixed_order_l_tuples[nu]]
            B_basis_concatenated_grad.append(
                torch.concatenate(gradients_nu).moveaxis(0, 2)
            )

        # Sum over like-atoms in each structure:
        sum_indices_comp = self.n_species*comp_metadata[:, 0]+self.species_to_species_index[comp_metadata[:, 2]]
        sum_indices_rs = self.n_species*rs_metadata[:, 0]+self.species_to_species_index[rs_metadata[:, 2]]
        sum_indices_spex = self.n_species*spex_metadata[:, 0]+self.species_to_species_index[spex_metadata[:, 2]]
        B_basis_per_structure = []
        for nu in range(0, self.nu_max+1):
            if nu == 0:
                B_basis_per_structure_nu = torch.zeros(
                    n_structures*self.n_species, B_basis_concatenated[nu].shape[1], dtype=torch.get_default_dtype(), device=self.device
                ).index_add_(0, sum_indices_comp, B_basis_concatenated[nu])
                B_basis_per_structure_nu = B_basis_per_structure_nu.reshape(n_structures, -1).sum(dim=1)
            elif nu == 1:
                B_basis_per_structure_nu = torch.zeros(
                    n_structures*self.n_species, B_basis_concatenated[nu].shape[1], dtype=torch.get_default_dtype(), device=self.device
                ).index_add_(0, sum_indices_rs, B_basis_concatenated[nu])
            else:
                B_basis_per_structure_nu = torch.zeros(
                    n_structures*self.n_species, B_basis_concatenated[nu].shape[1], dtype=torch.get_default_dtype(), device=self.device
                ).index_add_(0, sum_indices_spex, B_basis_concatenated[nu])
            B_basis_per_structure.append(
                B_basis_per_structure_nu.reshape(n_structures, -1)
            )

        n_force_centers = sum([structure.positions.shape[0] for structure in structures])
        structure_offsets = torch.tensor(np.cumsum([0]+[structure.positions.shape[0] for structure in structures[:-1]]), dtype=torch.long, device=self.device)
        sum_indices_comp_grad = self.n_species*(structure_offsets[comp_grad_metadata[:, 1]]+comp_grad_metadata[:, 2])+self.species_to_species_index[comp_metadata[comp_grad_metadata[:, 0], 2]]
        sum_indices_rs_grad = self.n_species*(structure_offsets[rs_grad_metadata[:, 1]]+rs_grad_metadata[:, 2])+self.species_to_species_index[rs_metadata[rs_grad_metadata[:, 0], 2]]
        sum_indices_spex_grad = self.n_species*(structure_offsets[spex_grad_metadata[:, 1]]+spex_grad_metadata[:, 2])+self.species_to_species_index[spex_metadata[spex_grad_metadata[:, 0], 2]]

        B_basis_per_structure_grad = []
        for nu in range(0, self.nu_max+1):
            if nu == 0:
                B_basis_per_structure_nu_grad = torch.zeros(
                    n_force_centers*self.n_species, 3, B_basis_concatenated_grad[nu].shape[2], dtype=torch.get_default_dtype(), device=self.device
                ).index_add_(0, sum_indices_comp_grad, B_basis_concatenated_grad[nu])
                B_basis_per_structure_nu_grad = B_basis_per_structure_nu_grad.reshape(n_force_centers, self.n_species, 3, -1)
            elif nu == 1:
                B_basis_per_structure_nu_grad = torch.zeros(
                    n_force_centers*self.n_species, 3, B_basis_concatenated_grad[nu].shape[2], dtype=torch.get_default_dtype(), device=self.device
                ).index_add_(0, sum_indices_rs_grad, B_basis_concatenated_grad[nu])
            else:
                B_basis_per_structure_nu_grad = torch.zeros(
                    n_force_centers*self.n_species, 3, B_basis_concatenated_grad[nu].shape[2], dtype=torch.get_default_dtype(), device=self.device
                ).index_add_(0, sum_indices_spex_grad, B_basis_concatenated_grad[nu])
            
            B_basis_per_structure_nu_grad = B_basis_per_structure_nu_grad.reshape(n_force_centers, self.n_species, 3, -1).swapaxes(1, 2).reshape(n_force_centers, 3, -1)
            if nu == 0: B_basis_per_structure_nu_grad = B_basis_per_structure_nu_grad.sum(dim=2, keepdim=True)
            B_basis_per_structure_grad.append(B_basis_per_structure_nu_grad)
        
        # Concatenate different body-orders:
        B_basis_all_together = torch.concatenate([B_basis_per_structure[nu] for nu in range(0, self.nu_max+1)], dim=1)
        B_basis_all_together_grad = torch.concatenate([B_basis_per_structure_grad[nu] for nu in range(0, self.nu_max+1)], dim=2)
        print("Number of features per body-order:", [B_basis_per_structure[nu].shape[1] for nu in range(0, self.nu_max+1)])

        return B_basis_all_together, B_basis_all_together_grad


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    np.random.seed(2)
    dictionary = {
        "r_cut_rs": 6.0,
        "r_cut": 6.0,
        "E_max": [1e+30, 1000.0, 25.0],
        "all_species": [1, 6],
        "le_type": "pure",
        "factor": 1.0,
        "factor2": 0.6,
        "cost_trade_off": False,
        "device": "cpu",
        "is_trace": False,
    }
    le_ace = LE_ACE(**dictionary)

    DATASET_PATH = "datasets/rmd17/benzene1.extxyz"
    train_slice = "5:10"
    test_slice = "1000:1005"

    import ase
    from ase import io

    train_structures = ase.io.read(DATASET_PATH, train_slice)
    test_structures = ase.io.read(DATASET_PATH, test_slice)

    train_values, train_gradients = le_ace.compute_with_gradients(train_structures)

    delta = 1e-5
    structure_counter = 0
    counter = 0
    for structure in train_structures:
        for i_atom in range(structure.positions.shape[0]):
            for alpha in range(3):
                train_structures[structure_counter].positions[i_atom, alpha] += delta
                train_values_plus = le_ace(train_structures)
                train_structures[structure_counter].positions[i_atom, alpha] -= 2.0*delta
                train_values_minus = le_ace(train_structures)
                train_structures[structure_counter].positions[i_atom, alpha] += delta
                numerical_derivative = (train_values_plus[structure_counter] - train_values_minus[structure_counter])/(2.0*delta)
                # print(numerical_derivative-train_gradients[counter][alpha])
                assert torch.allclose(numerical_derivative, train_gradients[counter][alpha])
            counter += 1
        structure_counter += 1
