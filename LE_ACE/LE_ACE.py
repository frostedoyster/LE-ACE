import numpy as np
import torch
import rascaline.torch
from .LE_initialization import initialize_basis
from .LE_metadata import get_le_metadata
from .generalized_cgs import get_generalized_cgs
from .ACE_calculator import ACECalculator
from .solver import Solver
from metatensor.torch import Labels

from .LE_expansion import get_LE_expansion
from .TRACE_expansion import get_TRACE_expansion
from .error_measures import get_rmse, get_mae, get_sae, get_sse


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
        fixed_stoichiometry,
        is_trace,
        n_trace,
        device
    ):
        super().__init__()

        nu_max = len(E_max) - 1
        self.E_max = E_max
        self.nu_max = nu_max
        self.device = device
        self.fixed_stoichiometry = fixed_stoichiometry

        self.all_species = [int(species) for species in all_species]
        self.n_species = len(all_species)
        self.all_species_tensor = torch.tensor(all_species, dtype=torch.long, device=self.device)
        self.species_to_species_index = torch.zeros(max(all_species)+1, dtype=torch.long, device=self.device)
        self.is_trace = is_trace
        if self.is_trace:
            self.n_trace = n_trace
            self.trace_comb = torch.tensor(np.random.normal(size=(self.n_species, n_trace)), device=device)

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

        self.combine_indices, self.multiplicities, self.LE_energies = get_le_metadata(all_species, E_max, n_max_l, E_ln, is_trace, n_trace, device)

        l_tuples = [0, 1] + [list(self.combine_indices[nu].keys()) for nu in range(2, nu_max+1)]
        requested_L_sigma_pairs = [(0, 1)]
        self.generalized_cgs = get_generalized_cgs(l_tuples, requested_L_sigma_pairs, nu_max, device)
        necessary_l_tuples_nu_max = []
        for requested_L_sigma_pair in requested_L_sigma_pairs:
            necessary_l_tuples_nu_max.extend(list(self.generalized_cgs[requested_L_sigma_pair][nu_max].keys()))

        # For correlation order nu_max, remove unncecessary indices that violate the parity constraints, so that the corresponding A-basis is never calculated:
        # FIXME: move somewhere else
        keys_to_pop = []
        for l_tuple_nu_max in self.combine_indices[nu_max].keys():
            if l_tuple_nu_max not in necessary_l_tuples_nu_max:
                keys_to_pop.append(l_tuple_nu_max)
        
        for l_tuple_nu_max in keys_to_pop:
            self.combine_indices[nu_max].pop(l_tuple_nu_max)
            self.multiplicities[nu_max].pop(l_tuple_nu_max)
            self.LE_energies[nu_max].pop(l_tuple_nu_max)

        # Needed to have consistent ordering when concatenating
        self.fixed_order_l_tuples = [0, 1] + [list(self.generalized_cgs[(0, 1)][nu].keys()) for nu in range(2, nu_max+1)]  
        # these, thanks to the ordering enforced in the metadata generation step, should only be the necessary generalized cgs
        # and they will be ordered lexicographically for each nu

        # Build extended LE_energies according to L_tuple and a_i:
        self.extended_LE_energies = []
        for nu in range(self.nu_max+1):
            if nu == 0:
                if fixed_stoichiometry:
                    lst = [0.0]
                else:
                    lst = [0.0]*self.n_species
                extended_LE_energies_nu = torch.tensor(lst, dtype=torch.get_default_dtype(), device=self.device)
            elif nu == 1:
                n_repeat = (self.n_species*self.n_trace if self.is_trace else self.n_species**2)
                extended_LE_energies_nu = torch.tile(
                    torch.tensor(self.E_n0[:self.n_max_rs], dtype=torch.get_default_dtype(), device=self.device),
                    (n_repeat,)
                )
            else:
                extended_LE_energies_nu = []
                for l_tuple in self.fixed_order_l_tuples[nu]:
                    extended_LE_energies_nu.append(
                        torch.tile(
                            self.LE_energies[nu][l_tuple],
                            (self.generalized_cgs[(0, 1)][nu][l_tuple].shape[0],)  # Account for different L
                        )
                    )
                extended_LE_energies_nu = torch.concatenate(extended_LE_energies_nu)
                extended_LE_energies_nu = torch.tile(
                    extended_LE_energies_nu,
                    (self.n_species,)  # Account for different a_i
                )
            self.extended_LE_energies.append(extended_LE_energies_nu)
        print([tensor.shape[0] for tensor in self.extended_LE_energies])

        self.nu0_calculator_train = rascaline.torch.AtomicComposition(per_structure=False)
        self.radial_spectrum_calculator_train = radial_spectrum_calculator
        self.spherical_expansion_calculator_train = spherical_expansion_calculator
        self.ace_calculator = ACECalculator(l_max, self.combine_indices, self.multiplicities, self.generalized_cgs)

    def compute_features(self, structures):
        n_structures = len(structures)
        structures = rascaline.torch.systems_to_torch(structures)

        composition_features_tmap = self.nu0_calculator_train.compute(structures)
        composition_features_tmap = composition_features_tmap.keys_to_samples("species_center")

        if self.is_trace:
            radial_spectrum_tmap = get_TRACE_expansion(structures, self.radial_spectrum_calculator_train, self.E_n0, self.E_max[1], self.all_species, self.trace_comb, rs=True, device=self.device)
            spherical_expansion_tmap = get_TRACE_expansion(structures, self.spherical_expansion_calculator_train, self.E_nl, self.E_max[2], self.all_species, self.trace_comb, device=self.device)
        else:
            radial_spectrum_tmap = get_LE_expansion(structures, self.radial_spectrum_calculator_train, self.E_n0, self.E_max[1], self.all_species, rs=True, device=self.device)
            spherical_expansion_tmap = get_LE_expansion(structures, self.spherical_expansion_calculator_train, self.E_nl, self.E_max[2], self.all_species, device=self.device)

        radial_spectrum_tmap = radial_spectrum_tmap.keys_to_samples("a_i")
        spherical_expansion_tmap = spherical_expansion_tmap.keys_to_samples("a_i")

        comp_metadata = composition_features_tmap.block(0).samples.values.to(self.device)
        rs_metadata = radial_spectrum_tmap.block(0).samples.values.to(self.device)
        spex_metadata = spherical_expansion_tmap.block(0).samples.values.to(self.device)

        composition_features = composition_features_tmap.block(0).values.to(self.device)
        radial_spectrum = radial_spectrum_tmap.block(0).values.to(self.device)  # NEEDED DUE TO BUG
        spherical_expansion = {(l,): block.values.swapaxes(0, 2) for (l,), block in spherical_expansion_tmap.items()}

        B_basis = self.ace_calculator(spherical_expansion)[(0, 1)]
        
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
                B_basis_per_structure_nu = B_basis_per_structure_nu.reshape(n_structures, -1)
                if self.fixed_stoichiometry:
                    B_basis_per_structure_nu = B_basis_per_structure_nu.sum(dim=1)
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

    def compute_features_with_gradients(self, structures):

        n_structures = len(structures)
        structures = rascaline.torch.systems_to_torch(structures)

        gradients = ["positions"]

        composition_features_tmap = self.nu0_calculator_train.compute(structures, gradients=gradients)
        composition_features_tmap = composition_features_tmap.keys_to_samples("species_center")

        if self.is_trace:
            radial_spectrum_tmap = get_TRACE_expansion(structures, self.radial_spectrum_calculator_train, self.E_n0, self.E_max[1], self.all_species, self.trace_comb, rs=True, do_gradients=True, device=self.device)
            spherical_expansion_tmap = get_TRACE_expansion(structures, self.spherical_expansion_calculator_train, self.E_nl, self.E_max[2], self.all_species, self.trace_comb, do_gradients=True, device=self.device)
        else:
            radial_spectrum_tmap = get_LE_expansion(structures, self.radial_spectrum_calculator_train, self.E_n0, self.E_max[1], self.all_species, rs=True, do_gradients=True, device=self.device)
            spherical_expansion_tmap = get_LE_expansion(structures, self.spherical_expansion_calculator_train, self.E_nl, self.E_max[2], self.all_species, do_gradients=True, device=self.device)

        radial_spectrum_tmap = radial_spectrum_tmap.keys_to_samples("a_i")
        spherical_expansion_tmap = spherical_expansion_tmap.keys_to_samples("a_i")

        comp_metadata = composition_features_tmap.block(0).samples.values.to(self.device)
        rs_metadata = radial_spectrum_tmap.block(0).samples.values.to(self.device)
        spex_metadata = spherical_expansion_tmap.block(0).samples.values.to(self.device)

        composition_features = composition_features_tmap.block(0).values.to(self.device)
        radial_spectrum = radial_spectrum_tmap.block(0).values.to(self.device)  # DUE TO BUG
        spherical_expansion = {(l,): block.values.swapaxes(0, 2) for (l,), block in spherical_expansion_tmap.items()}

        comp_grad_metadata = composition_features_tmap.block(0).gradient("positions").samples.values.to(self.device)
        rs_grad_metadata = radial_spectrum_tmap.block(0).gradient("positions").samples.values.to(self.device)
        spex_grad_metadata = spherical_expansion_tmap.block(0).gradient("positions").samples.values.to(self.device)

        composition_features_grad = composition_features_tmap.block(0).gradient("positions").values.to(self.device)
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

        B_basis, B_basis_gradients = self.ace_calculator.compute_with_gradients(spherical_expansion, spherical_expansion_grad, spex_grad_metadata[:, 0])
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
                B_basis_per_structure_nu = B_basis_per_structure_nu.reshape(n_structures, -1)
                if self.fixed_stoichiometry:
                    # Sum contributions from all atoms
                    B_basis_per_structure_nu = B_basis_per_structure_nu.sum(dim=1)
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
            if nu == 0 and self.fixed_stoichiometry:
                B_basis_per_structure_nu_grad = B_basis_per_structure_nu_grad.sum(dim=2, keepdim=True)
            B_basis_per_structure_grad.append(B_basis_per_structure_nu_grad)
        
        # Concatenate different body-orders:
        B_basis_all_together = torch.concatenate([B_basis_per_structure[nu] for nu in range(0, self.nu_max+1)], dim=1)
        B_basis_all_together_grad = torch.concatenate([B_basis_per_structure_grad[nu] for nu in range(0, self.nu_max+1)], dim=2)
        print("Number of features per body-order:", [B_basis_per_structure[nu].shape[1] for nu in range(0, self.nu_max+1)])

        return B_basis_all_together, B_basis_all_together_grad


    def forward():
        # Allows training with backpropagation
        raise NotImplementedError()


    def to_predict():

        # Prepare the LE_ACE_predict class
        raise NotImplementedError()

        return le_ace_predict


    def train(self, train_structures, validation_structures=None, test_structures=None, training_style="linear algebra", backend="rascaline", target_key="energy", batch_size=None, do_gradients=False, force_weight=0.03, opt_target_name="rmse"):

        n_train = len(train_structures)

        if validation_structures is None:
            np.random.shuffle(train_structures)
            n_validation = n_train // 10
            n_train = n_train - n_validation
            validation_structures = train_structures[:n_validation]
            train_structures = train_structures[n_validation:]
        else:
            n_validation = len(validation_structures)

        if test_structures is not None: n_test = len(test_structures)

        if batch_size is None:
            batch_size = 10 if do_gradients else 100

        train_energies = torch.tensor([structure.info[target_key] for structure in train_structures], dtype=torch.get_default_dtype(), device=self.device)
        validation_energies = torch.tensor([structure.info[target_key] for structure in validation_structures], dtype=torch.get_default_dtype(), device=self.device)
        if test_structures is not None: 
            test_energies = torch.tensor([structure.info[target_key] for structure in test_structures], dtype=torch.get_default_dtype(), device=self.device)

        if do_gradients:
            train_forces = torch.tensor(np.concatenate([structure.get_forces() for structure in train_structures], axis = 0), dtype=torch.get_default_dtype(), device=self.device)*force_weight
            validation_forces = torch.tensor(np.concatenate([structure.get_forces() for structure in validation_structures], axis = 0), dtype=torch.get_default_dtype(), device=self.device)*force_weight
            if test_structures is not None:
                test_forces = torch.tensor(np.concatenate([structure.get_forces() for structure in test_structures], axis = 0), dtype=torch.get_default_dtype(), device=self.device)*force_weight

        # Divide training and test set into batches (to limit memory usage):
        def get_batches(list: list, batch_size: int) -> list:
            batches = []
            n_full_batches = len(list)//batch_size
            for i_batch in range(n_full_batches):
                batches.append(list[i_batch*batch_size:(i_batch+1)*batch_size])
            if len(list) % batch_size != 0:
                batches.append(list[n_full_batches*batch_size:])
            return batches

        train_structures = get_batches(train_structures, batch_size)
        validation_structures = get_batches(validation_structures, batch_size)
        if test_structures is not None: validation_structures = get_batches(test_structures, batch_size)

        X_train_batches = []
        X_train_batches_grad = []
        for i_batch, batch in enumerate(train_structures):
            print(f"DOING TRAIN BATCH {i_batch+1} out of {len(train_structures)}")
            if do_gradients:
                values, gradients = self.compute_features_with_gradients(batch)
                gradients = -force_weight*gradients.reshape(gradients.shape[0]*3, values.shape[1])
                X_train_batches.append(values)
                X_train_batches_grad.append(gradients)
            else:
                values = self.compute_features(batch)
                X_train_batches.append(values)

        if do_gradients:
            X_train = torch.concat(X_train_batches + X_train_batches_grad, dim = 0)
        else:
            X_train = torch.concat(X_train_batches, dim = 0)

        X_validation_batches = []
        X_validation_batches_grad = []
        for i_batch, batch in enumerate(validation_structures):
            print(f"DOING VALIDATION BATCH {i_batch+1} out of {len(validation_structures)}")
            if do_gradients:
                values, gradients = self.compute_features_with_gradients(batch)
                gradients = -force_weight*gradients.reshape(gradients.shape[0]*3, values.shape[1])
                X_validation_batches.append(values)
                X_validation_batches_grad.append(gradients)
            else:
                values = self.compute_features(batch)
                X_validation_batches.append(values)

        if do_gradients:
            X_validation = torch.concat(X_validation_batches + X_validation_batches_grad, dim = 0)
        else:
            X_validation = torch.concat(X_validation_batches, dim = 0)

        if test_structures is not None:
            X_test_batches = []
            X_test_batches_grad = []
            for i_batch, batch in enumerate(test_structures):
                print(f"DOING TEST BATCH {i_batch+1} out of {len(test_structures)}")
                if do_gradients:
                    values, gradients = self.compute_features_with_gradients(batch)
                    gradients = -force_weight*gradients.reshape(gradients.shape[0]*3, values.shape[1])
                    X_test_batches.append(values)
                    X_test_batches_grad.append(gradients)
                else:
                    values = self.compute_features(batch)
                    X_test_batches.append(values)

            if do_gradients:
                X_test = torch.concat(X_test_batches + X_test_batches_grad, dim = 0)
            else:
                X_test = torch.concat(X_test_batches, dim = 0)

        print("Features done")

        print("Beginning linear fit optimization")

        if do_gradients:
            train_targets = torch.concat([train_energies, train_forces.reshape((-1,))])
            validation_targets = torch.concat([validation_energies, validation_forces.reshape((-1,))])
            if test_structures is not None: test_targets = torch.concat([test_energies, test_forces.reshape((-1,))])
        else:
            train_targets = train_energies
            validation_targets = validation_energies
            if test_structures is not None:
                test_targets = test_energies

        symm = X_train.T @ X_train
        vec = X_train.T @ train_targets
        n_feat = X_train.shape[1]
        print("Total number of features: ", n_feat)

        alpha_start = -10.0
        beta_start = 0.0

        solver = Solver(n_feat, self.extended_LE_energies, alpha_start, beta_start, self.nu_max).to(self.device)
        optimizer = torch.optim.LBFGS(solver.parameters(), max_iter=5)

        loss_list = []
        alpha_list = []
        beta_list = []

        def closure():
            optimizer.zero_grad()

            c = solver(symm, vec)
            validation_predictions = X_validation @ c

            if opt_target_name == "mae":
                loss = get_sae(validation_predictions, validation_targets)
            else:
                loss = get_sse(validation_predictions, validation_targets)
            
            print(f"alpha={solver.alpha.item()} beta={solver.beta.item()} loss={loss.item()}")
            loss_list.append(loss.item())
            alpha_list.append(solver.alpha.item())
            beta_list.append(solver.beta.item())

            loss.backward()
            return loss


        n_cycles = 4
        for i_cycle in range(n_cycles):
            _ = optimizer.step(closure)
            print(f"Finished step {i_cycle+1} out of {n_cycles}")

        where_best_loss = np.argmin(np.nan_to_num(loss_list, nan=1e100))
        best_alpha = alpha_list[where_best_loss]
        best_beta = beta_list[where_best_loss]
        print("Best parameters:", best_alpha, best_beta)

        LE_reg = [tensor.clone() for tensor in self.extended_LE_energies]
        for nu in range(self.nu_max+1):
            LE_reg[nu] *= np.exp(best_beta*nu)
        LE_reg = torch.concatenate(LE_reg)  

        for i in range(n_feat):
            symm[i, i] += 10**best_alpha*LE_reg[i]
        c = torch.linalg.solve(symm, vec)

        validation_predictions = X_validation @ c
        print("n_train:", n_train, "n_features:", n_feat)
        print(f"Validation set RMSE (E): {get_rmse(validation_predictions[:n_validation], validation_targets[:n_validation]).item()} [MAE (E): {get_mae(validation_predictions[:n_validation], validation_targets[:n_validation]).item()}], RMSE (F): {get_rmse(validation_predictions[n_validation:], validation_targets[n_validation:]).item()/force_weight} [MAE (F): {get_mae(validation_predictions[n_validation:], validation_targets[n_validation:]).item()/force_weight}]")

        if test_structures is not None:
            validation_predictions = X_validation @ c
            print(f"Test set RMSE (E): {get_rmse(test_predictions[:n_test], test_targets[:n_test]).item()} [MAE (E): {get_mae(test_predictions[:n_test], test_targets[:n_test]).item()}], RMSE (F): {get_rmse(test_predictions[n_test:], test_targets[n_test:]).item()/force_weight} [MAE (F): {get_mae(test_predictions[n_test:], test_targets[n_test:]).item()/force_weight}]")

        return {
            "validation RMSE energies": get_rmse(validation_predictions[:n_validation], validation_targets[:n_validation]).item(),
            "validation MAE energies": get_mae(validation_predictions[:n_validation], validation_targets[:n_validation]).item(),
            "validation RMSE forces": get_rmse(validation_predictions[n_validation:], validation_targets[n_validation:]).item()/force_weight if do_gradients else None,
            "validation MAE forces": get_mae(validation_predictions[n_validation:], validation_targets[n_validation:]).item()/force_weight if do_gradients else None,
        }
