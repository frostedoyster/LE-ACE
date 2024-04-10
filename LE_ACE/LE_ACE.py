import numpy as np
import torch
import rascaline.torch
from metatensor.torch import Labels

from .LE_initialization import initialize_basis
from .LE_metadata import get_le_metadata
from .generalized_cgs import get_generalized_cgs
from .ACE_calculator import ACECalculator
from .solver import Solver
from .structures import transform_structures
from .ACE_evaluator import ACEEvaluator

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
        self.is_trace = is_trace
        if self.is_trace:
            self.n_trace = n_trace
            self.trace_comb = torch.tensor(np.random.normal(size=(self.n_species, n_trace)), device=device)

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
        self.n_max_l = []
        for l in range(l_max+1):
            self.n_max_l.append(np.where(self.E_nl[:, l] <= E_max[2])[0][-1] + 1)
        print("Radial spectrum:", self.n_max_rs)
        print("Spherical expansion", self.n_max_l)
        self.l_max = len(self.n_max_l) - 1
        E_ln = [E_ln[l][:self.n_max_l[l]] for l in range(l_max+1)]

        self.combine_indices, self.multiplicities, self.LE_energies = get_le_metadata(all_species, E_max, self.n_max_l, E_ln, is_trace, n_trace, device)

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
        print("Number of features per body-order:", [tensor.shape[0] for tensor in self.extended_LE_energies])

        # Dummy properties, to be changed: FIXME
        self.properties_per_atom = (
            [Labels.range("property", 1)]
            +
            [Labels.range("property", self.n_max_rs*(self.n_trace if self.is_trace else self.n_species))]
            +
            [
                Labels.range(
                    "property",
                    sum([self.generalized_cgs[(0, 1)][nu][l_tuple].shape[0] for l_tuple in self.fixed_order_l_tuples[nu]])
                )
                for nu in range(2, nu_max+1)
            ]
        )
        self.properties_per_structure = [Labels.range("property", len(LE_energies)) for LE_energies in self.extended_LE_energies]

        self.nu0_calculator_train = rascaline.torch.AtomicComposition(per_system=False)
        self.radial_spectrum_calculator_train = radial_spectrum_calculator
        self.spherical_expansion_calculator_train = spherical_expansion_calculator
        self.ace_calculator = ACECalculator(l_max, self.combine_indices, self.multiplicities, self.generalized_cgs)

        self.prediction_coefficients = None  # to be set during training

    def compute_features(self, structures):

        # transform if not already done
        if not isinstance(structures[0], torch.ScriptObject):
            structures = transform_structures(structures)

        n_structures = len(structures)

        composition_features_tmap = self.nu0_calculator_train.compute(structures)
        composition_features_tmap = composition_features_tmap.keys_to_samples("center_type")

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
        spherical_expansion = {(key.values[0].item(),): block.values.swapaxes(0, 2) for key, block in spherical_expansion_tmap.items()}

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
        B_basis_all_together = torch.concatenate([B_basis_per_structure[nu] for nu in range(0, self.nu_max+1)], dim=1)

        return B_basis_all_together

    def compute_features_with_gradients(self, structures):

        n_structures = len(structures)
        
        # transform if not already done
        if not isinstance(structures[0], torch.ScriptObject):
            structures = transform_structures(structures)

        gradients = ["positions"]

        composition_features_tmap = self.nu0_calculator_train.compute(structures, gradients=gradients)
        composition_features_tmap = composition_features_tmap.keys_to_samples("center_type")

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
        spherical_expansion = {(key.values[0].item(),): block.values.swapaxes(0, 2) for key, block in spherical_expansion_tmap.items()}

        comp_grad_metadata = composition_features_tmap.block(0).gradient("positions").samples.values.to(self.device)
        rs_grad_metadata = radial_spectrum_tmap.block(0).gradient("positions").samples.values.to(self.device)
        spex_grad_metadata = spherical_expansion_tmap.block(0).gradient("positions").samples.values.to(self.device)

        composition_features_grad = composition_features_tmap.block(0).gradient("positions").values.to(self.device)
        radial_spectrum_grad = radial_spectrum_tmap.block(0).gradient("positions").values.to(self.device)  # DUE TO BUG
        spherical_expansion_grad = {
            (key.values[0].item(),): block.gradient("positions").values.reshape(-1, block.gradient("positions").values.shape[2], block.gradient("positions").values.shape[3]).swapaxes(0, 2).reshape(
                block.gradient("positions").values.shape[3],
                block.gradient("positions").values.shape[2],
                block.gradient("positions").values.shape[0],
                block.gradient("positions").values.shape[1]
            )
            for key, block in spherical_expansion_tmap.items()
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

        return B_basis_all_together, B_basis_all_together_grad

    def forward():
        # Allows training with backpropagation
        raise NotImplementedError()

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

        if batch_size is None:
            batch_size = 10 if do_gradients else 100

        # Divide training and test set into batches to limit memory usage:
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

        n_feat = sum(tensor.shape[0] for tensor in self.extended_LE_energies)
        symm = torch.zeros((n_feat, n_feat), dtype=torch.get_default_dtype(), device=self.device)
        vec = torch.zeros((n_feat,), dtype=torch.get_default_dtype(), device=self.device)

        for i_batch, batch in enumerate(train_structures):
            print(f"DOING TRAIN BATCH {i_batch+1} out of {len(train_structures)}")
            energies = torch.tensor([structure.info[target_key] for structure in batch], dtype=torch.get_default_dtype(), device=self.device)
            if do_gradients:
                values, gradients = self.compute_features_with_gradients(batch)
                gradients = -force_weight*gradients.reshape(gradients.shape[0]*3, values.shape[1])
                symm += values.T @ values
                symm += gradients.T @ gradients
                forces = torch.tensor(np.concatenate([structure.get_forces().reshape(-1) for structure in batch], axis = 0), dtype=torch.get_default_dtype(), device=self.device)*force_weight
                vec += values.T @ energies
                vec += gradients.T @ forces
            else:
                values = self.compute_features(batch)
                symm += values.T @ values
                vec += values.T @ energies

        print("Features done. Beginning linear fit optimization")
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
            # batched LBFGS
            total_loss = 0.0
            for batch in validation_structures:
                c = solver(symm, vec)
                self.prediction_coefficients = c
                predictions_dict = self.predict(batch, do_positions_grad=do_gradients, is_training=True)
                validation_predictions = predictions_dict["values"]
                validation_targets = torch.tensor([structure.info[target_key] for structure in batch], dtype=torch.get_default_dtype(), device=self.device)
                if do_gradients:
                    validation_predictions_forces = -predictions_dict["positions gradient"].reshape(-1)*force_weight
                    validation_targets_forces = torch.tensor(np.concatenate([structure.get_forces().reshape(-1) for structure in batch], axis = 0), dtype=torch.get_default_dtype(), device=self.device)*force_weight
                    validation_predictions = torch.cat((validation_predictions, validation_predictions_forces))
                    validation_targets = torch.cat((validation_targets, validation_targets_forces))

                if opt_target_name == "mae":
                    loss = get_sae(validation_predictions, validation_targets)
                else:
                    loss = get_sse(validation_predictions, validation_targets)

                total_loss += loss.item()
                loss.backward()
            
            print(f"alpha={solver.alpha.item()} beta={solver.beta.item()} loss={total_loss}")
            loss_list.append(total_loss)
            alpha_list.append(solver.alpha.item())
            beta_list.append(solver.beta.item())

            return total_loss

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
        # register prediction coefficients for evaluation
        self.prediction_coefficients = c

        energy_predictions = []
        energy_targets = []
        if do_gradients:
            force_predictions = []
            force_targets = []
        for batch in validation_structures:
            predict_dict = self.predict(batch, do_positions_grad=do_gradients)
            energy_predictions.append(predict_dict["values"])
            energy_targets.append(torch.tensor([structure.info[target_key] for structure in batch], dtype=torch.get_default_dtype(), device=self.device))
            if do_gradients:
                force_predictions.append(-predict_dict["positions gradient"])
                force_targets.append(torch.tensor(np.concatenate([structure.get_forces() for structure in batch], axis = 0), dtype=torch.get_default_dtype(), device=self.device))
        energy_predictions = torch.concatenate(energy_predictions)
        energy_targets = torch.concatenate(energy_targets)
        if do_gradients:
            force_predictions = torch.concatenate(force_predictions)
            force_targets = torch.concatenate(force_targets)


        validation_rmse_energies = get_rmse(energy_predictions, energy_targets).item()
        validation_mae_energies = get_mae(energy_predictions, energy_targets).item()
        if do_gradients:
            validation_rmse_forces = get_rmse(force_predictions, force_targets).item()
            validation_mae_forces = get_mae(force_predictions, force_targets).item()
        else:
            validation_rmse_forces = None
            validation_mae_forces = None 
        print(f"Validation set RMSE (E): {validation_rmse_energies} [MAE (E): {validation_mae_energies}], RMSE (F): {validation_rmse_forces} [MAE (F): {validation_mae_forces}]")

        return_dict = {
            "validation RMSE energies": validation_rmse_energies,
            "validation MAE energies": validation_mae_energies,
        }
        if do_gradients:
            return_dict["validation RMSE forces"] = validation_rmse_forces
            return_dict["validation MAE forces"] = validation_mae_forces

        # Now, if requested, we test on the test set:
        if test_structures is not None:
            test_structures = get_batches(test_structures, batch_size)
            energy_predictions = []
            energy_targets = []
            if do_gradients:
                force_predictions = []
                force_targets = []
            for batch in test_structures:
                predict_dict = self.predict(batch, do_positions_grad=do_gradients)
                energy_predictions.append(predict_dict["values"])
                energy_targets.append(torch.tensor([structure.info[target_key] for structure in batch], dtype=torch.get_default_dtype(), device=self.device))
                if do_gradients:
                    force_predictions.append(-predict_dict["positions gradient"])
                    force_targets.append(torch.tensor(np.concatenate([structure.get_forces() for structure in batch], axis = 0), dtype=torch.get_default_dtype(), device=self.device))
            energy_predictions = torch.concatenate(energy_predictions)
            energy_targets = torch.concatenate(energy_targets)
            if do_gradients:
                force_predictions = torch.concatenate(force_predictions)
                force_targets = torch.concatenate(force_targets)

            test_rmse_energies = get_rmse(energy_predictions, energy_targets).item()
            test_mae_energies = get_mae(energy_predictions, energy_targets).item()
            if do_gradients:
                test_rmse_forces = get_rmse(force_predictions, force_targets).item()
                test_mae_forces = get_mae(force_predictions, force_targets).item()

            print(f"Test set RMSE (E): {test_rmse_energies} [MAE (E): {test_mae_energies}], RMSE (F): {test_rmse_forces} [MAE (F): {test_mae_forces}]")

            return_dict["test RMSE energies"] = test_rmse_energies
            return_dict["test MAE energies"] = test_mae_energies
            if do_gradients:
                return_dict["test RMSE forces"] = test_rmse_forces
                return_dict["test MAE forces"] = test_mae_forces

        return return_dict

    def predict(self, structures, do_positions_grad=False, do_cells_grad=False, is_training=False):
        # A relatively slow predictor implementation
        structures = transform_structures(structures, positions_requires_grad=do_positions_grad, cells_requires_grad=do_cells_grad)
        features = self.compute_features(structures)
        predictions = features @ self.prediction_coefficients

        predictions_dict = {
            "values": predictions
        }

        # !!!!!!!! Only works for the prediction of scalars.
        # torch.autograd.jacobian might be usable to do higher-order tensors
        if do_positions_grad:
            positions_grad = torch.autograd.grad(
                outputs=predictions,
                inputs=[structure.positions for structure in structures],
                grad_outputs=torch.ones_like(predictions),
                retain_graph=is_training,
                create_graph=is_training,
            )
            predictions_dict["positions gradient"] = torch.concatenate(positions_grad).to(self.device)  # These are always created on CPU

        if do_cells_grad:
            raise NotImplementedError()

        return predictions_dict

    def get_fast_evaluator(self):

        if self.prediction_coefficients is None:
            raise ValueError("The model has not been trained yet.")

        if self.is_trace:
            raise NotImplementedError

        # Split prediction coefficients according to body-order:
        body_order_split_sizes = [tensor.shape[0] for tensor in self.extended_LE_energies]

        split_nu_coefficients = torch.split(self.prediction_coefficients, body_order_split_sizes)

        for split_coefficients_single_nu in split_nu_coefficients[1:]:
            assert split_coefficients_single_nu.shape[0] % self.n_species == 0
        split_nu_ai_coefficients = [0] + [torch.split(split_coefficients_single_nu, split_coefficients_single_nu.shape[0]//self.n_species) for split_coefficients_single_nu in split_nu_coefficients[1:]]

        split_coefficients = [split_coefficients_single_nu[0]]
        split_coefficients.append([])
        for ai_index in range(self.n_species):
            split_coefficients[1].append([])
            split_coefficients[1][ai_index].append(split_nu_ai_coefficients[1][ai_index])
        for nu in range(2, self.nu_max+1):
            split_coefficients.append([])
            for ai_index in range(self.n_species):
                split_coefficients[nu].append(
                    torch.split(
                        split_nu_ai_coefficients[nu][ai_index],
                        [self.generalized_cgs[(0, 1)][nu][l_tuple].shape[0]*self.multiplicities[nu][l_tuple].shape[0] for l_tuple in self.fixed_order_l_tuples[nu]]
                    )
                )


        # REVERSEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE? Is this needed? 
        combine_indices_nu1 = [0, 1]
        for nu in range(2, self.nu_max+1):
            combine_indices_nu1.append([])
            for l_tuple_nu in self.fixed_order_l_tuples[nu]:
                combine_indices_nu1_l_tuple_nu = []
                previous_iota_indices = torch.arange(0, self.combine_indices[nu][l_tuple_nu][1].shape[0], dtype=torch.long)
                current_tuple = l_tuple_nu
                for iota in range(nu, 1, -1):
                    current_combine_indices = self.combine_indices[iota][current_tuple]
                    combine_indices_nu1_l_tuple_nu.append(
                        current_combine_indices[1][previous_iota_indices]
                    )
                    if iota == 2:
                        combine_indices_nu1_l_tuple_nu.append(
                            current_combine_indices[0][previous_iota_indices]
                        )
                    else:
                        previous_iota_indices = current_combine_indices[0][previous_iota_indices]
                    current_tuple = current_tuple[:-1]

                combine_indices_nu1_l_tuple_nu.reverse()
                combine_indices_nu1[nu].append(
                    torch.stack(combine_indices_nu1_l_tuple_nu, dim=1)
                )

        multipliers = [
            self.prediction_coefficients[0],
            torch.stack([coefficients_1_ai for coefficients_1_ai in split_nu_ai_coefficients[1]])    
        ]
        for nu in range(2, self.nu_max+1):
            multipliers_nu = []
            for ai_index in range(self.n_species):
                multipliers_nu_ai = []
                for index_l_tuple_nu, l_tuple_nu in enumerate(self.fixed_order_l_tuples[nu]):
                    C = self.generalized_cgs[(0, 1)][nu][l_tuple_nu]  # m, L_tuple
                    c = split_coefficients[nu][ai_index][index_l_tuple_nu]  # L_tuple, b
                    m = self.multiplicities[nu][l_tuple_nu]  # b
                    multipliers_l_tuple = (C.T @ c.reshape(C.shape[0], m.shape[0])) * m.unsqueeze(0)
                    multipliers_nu_ai.append(multipliers_l_tuple.flatten())  # m_tuple, b
                multipliers_nu_ai = torch.concatenate(multipliers_nu_ai)
                multipliers_nu.append(multipliers_nu_ai)
            multipliers_nu = torch.stack(multipliers_nu)
            multipliers.append(multipliers_nu)

        l_lengths = torch.tensor(
            [self.n_species * self.n_max_l[l] for l in range(self.l_max+1)]
        )
        l_shifts = torch.cumsum(
            torch.tensor(
                [0] + [self.n_species * self.n_max_l[l] * (2*l+1) for l in range(self.l_max+1)]
            ),
            0
        )

        final_combine_indices = [0, 1]
        for nu in range(2, self.nu_max+1):
            final_combine_indices.append([])
            for index_l_tuple_nu, l_tuple_nu in enumerate(self.fixed_order_l_tuples[nu]):
                l_1 = l_tuple_nu[0]
                current_m_range = torch.arange(2*l_1+1, dtype=torch.int).reshape(-1, 1)
                for iota in range(1, nu):
                    new_l = l_tuple_nu[iota]
                    new_m_range = torch.arange(2*new_l+1, dtype=torch.int)
                    current_m_range = torch.concatenate(
                        [
                            torch.repeat_interleave(current_m_range, len(new_m_range), dim=0),
                            new_m_range.repeat(len(current_m_range)).reshape(-1, 1)
                        ],
                        dim=1
                    )
                m_tuples = torch.repeat_interleave(current_m_range, len(combine_indices_nu1[nu][index_l_tuple_nu]), dim=0)
                b_1 = combine_indices_nu1[nu][index_l_tuple_nu].repeat((len(current_m_range), 1))
                assert m_tuples.shape == b_1.shape
                final_combine_indices_l_tuple_nu = torch.empty_like(b_1)
                for iota in range(nu):
                    final_combine_indices_l_tuple_nu[:, iota] = l_shifts[l_tuple_nu[iota]] + l_lengths[l_tuple_nu[iota]] * m_tuples[:, iota] + b_1[:, iota]
                final_combine_indices[nu].append(final_combine_indices_l_tuple_nu)
            final_combine_indices[nu] = torch.concatenate(final_combine_indices[nu])

        for nu in range(2, self.nu_max+1):
            assert multipliers[nu].shape[1] == final_combine_indices[nu].shape[0]
            where_nonzero = torch.where(torch.sum((multipliers[nu] != 0.0).to(torch.int), dim=0) != 0)[0]  # at least one non-zero element
            #multipliers[nu] = multipliers[nu][:, where_nonzero]
            #final_combine_indices[nu] = final_combine_indices[nu][where_nonzero]
            print("nu", nu, ", number of polynomials", final_combine_indices[nu].shape[0])

        return ACEEvaluator(
            all_species=self.all_species,
            composition_calculator=self.nu0_calculator_train,
            radial_spectrum_calculator=self.radial_spectrum_calculator_train,
            spherical_expansion_calculator=self.spherical_expansion_calculator_train,
            combine_indices=final_combine_indices,
            multipliers=multipliers,
            l_max=self.l_max,
            E_nl=self.E_nl,
            E_n0=self.E_n0,
            E_max=self.E_max,
            device=self.device
        )
