import torch
import numpy as np

from equistore import TensorMap, Labels, TensorBlock

class TRACEInvariantCalculator(torch.nn.Module):

    def __init__(self, E_nl, combined_nl, all_species):
        super(TRACEInvariantCalculator, self).__init__()

        # Initialize with information that comes from nu = 1 and which is useful at every iteration:
        self.E_nl = E_nl
        self.combined_nl = combined_nl
        self.all_species = all_species

        self.nu_plus_one_count = {}
        self.properties_values = {}
        self.selected_features = {}

    def forward(self, TRACE_nu, TRACE_1, E_max_nu_plus_one):

        do_gradients = TRACE_1.block(0).has_gradient("positions")

        l_max = 0
        for idx, block in TRACE_1.items():
            l_max = max(l_max, idx["lam"])

        lam_max = 0
        for idx, block in TRACE_nu.items():
            lam_max = max(lam_max, idx["lam"])

        max_l_lam = min(l_max, lam_max)

        # Infer nu from length of history indices:
        nu = (len(TRACE_nu.block(0).properties.names)-1)//3  # remove k index and divide by 3

        keys = []
        blocks = []

        properties_names = (
            [f"{name}" for name in TRACE_nu.block(0).properties.names]
            + [f"{name[:-1]}{nu+1}" for name in TRACE_1.block(0).properties.names[1:]]
        )  # Add l_nu, n_nu, k_nu

        for a_i in self.all_species:
            print(f"Combining center species = {a_i}")

            if nu not in self.nu_plus_one_count:

                nu_plus_one_count = 0
                selected_features = {}
                properties_values = []

                for l in range(max_l_lam+1):  # l and lambda are now the same thing
                    selected_features[l] = []

                    block_nu = TRACE_nu.block(lam=l, a_i=a_i)
                    z = block_nu.properties["z"]
                    n_nu = block_nu.properties["n"+str(nu)]
                    l_nu = block_nu.properties["l"+str(nu)]

                    block_1 = TRACE_1.block(lam=l, a_i=a_i)
                    z_nu_plus_1 = block_1.properties["z"]
                    n_nu_plus_1 = block_1.properties["n1"]

                    for q_nu in range(block_nu.values.shape[-1]):
                        for q_1 in range(block_1.values.shape[-1]):
                            # Check TRACE condition:
                            if z[q_nu] != z_nu_plus_1[q_1]: continue
                            # Check lexicographic ordering:
                            if self.combined_nl[(n_nu[q_nu], l_nu[q_nu])] > self.combined_nl[(n_nu_plus_1[q_1], l)]: continue  
                            # Check TRACE eigenvalue:
                            hypothetical_eigenvalue = 0.0
                            for nu in range(1, nu+1):
                                hypothetical_eigenvalue += self.E_nl[block_nu.properties["n"+str(nu)][q_nu], block_nu.properties["l"+str(nu)][q_nu]]
                            hypothetical_eigenvalue += self.E_nl[n_nu_plus_1[q_1], l]
                            if hypothetical_eigenvalue > E_max_nu_plus_one: continue 
                            # Check parity:
                            parity_checker = 0
                            for nu in range(1, nu+1):
                                parity_checker += block_nu.properties["l"+str(nu)][q_nu]
                            parity_checker += l
                            if parity_checker % 2 != 0: continue 

                            properties_list = [[block_nu.properties[name][q_nu] for name in block_nu.properties.names] + [block_1.properties[name][q_1] for name in block_1.properties.names[1:-1]] + [0]]
                            properties_values.append(properties_list)
                            selected_features[l].append([q_nu, q_1])
                            
                            nu_plus_one_count += 1

                keys_to_be_removed = []
                for key in selected_features.keys():
                    if len(selected_features[key]) == 0: 
                        keys_to_be_removed.append(key)  # No features were selected.
                    else:
                        selected_features[key] = torch.tensor(selected_features[key])

                for key in keys_to_be_removed:
                    selected_features.pop(key)

                self.nu_plus_one_count[nu] = nu_plus_one_count
                self.selected_features[nu] = selected_features
                self.properties_values[nu] = properties_values

            nu_plus_one_count = self.nu_plus_one_count[nu]
            selected_features = self.selected_features[nu]
            properties_values = self.properties_values[nu]

            block_1 = TRACE_1.block(lam=0, a_i=a_i)
            data = torch.empty((len(block_1.samples), nu_plus_one_count), device=block_1.values.device)
            if do_gradients: gradient_data = torch.zeros((len(block_1.gradient("positions").samples), 3, nu_plus_one_count), device=block_1.values.device)

            nu_plus_one_count = 0  # reset counter
            for l in range(max_l_lam+1):  # l and lambda are now the same thing
                if l not in selected_features: continue  # No features are selected.

                cg = 1.0/np.sqrt(2*l+1)

                block_nu = TRACE_nu.block(lam=l, a_i=a_i)
                if do_gradients: 
                    gradients_nu = block_nu.gradient("positions")
                    samples_for_gradients_nu = torch.tensor(gradients_nu.samples["sample"], dtype=torch.int64)

                block_1 = TRACE_1.block(lam=l, a_i=a_i)
                if do_gradients: 
                    gradients_1 = block_1.gradient("positions")
                    samples_for_gradients_1 = torch.tensor(gradients_1.samples["sample"], dtype=torch.int64)

                data[:, nu_plus_one_count:nu_plus_one_count+selected_features[l].shape[0]] = cg*torch.sum(block_nu.values[:, :, selected_features[l][:, 0]]*block_1.values[:, :, selected_features[l][:, 1]], dim = 1, keepdim = False)
                if do_gradients: gradient_data[:, :, nu_plus_one_count:nu_plus_one_count+selected_features[l].shape[0]] = cg * torch.sum(gradients_nu.values[:, :, :, selected_features[l][:, 0]] * block_1.values[samples_for_gradients_nu][:, :, selected_features[l][:, 1]].unsqueeze(dim=1) + block_nu.values[samples_for_gradients_1][:, :, selected_features[l][:, 0]].unsqueeze(dim=1) * gradients_1.values[:, :, :, selected_features[l][:, 1]], dim = 2, keepdim = False)  # exploiting broadcasting rules
                
                nu_plus_one_count += selected_features[l].shape[0]

            block = TensorBlock(
                values=data,
                samples=block_1.samples,
                components=[],
                properties=Labels(
                    names=properties_names,
                    values=np.asarray(np.vstack(properties_values), dtype=np.int32),
                ),
            )
            if do_gradients: block.add_gradient(
                parameter="positions",
                gradient=TensorBlock(
                    values = gradient_data, 
                    samples = gradients_1.samples, 
                    components = [gradients_1.components[0]],
                    properties = block.properties
                )
            )
            keys.append([a_i])
            blocks.append(block)

        TRACE_invariants = TensorMap(
            keys = Labels(
                names = ("a_i",),
                values = np.array(keys), # .reshape((-1, 2)),
            ), 
            blocks = blocks)

        return TRACE_invariants
