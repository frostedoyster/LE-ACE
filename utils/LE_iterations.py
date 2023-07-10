import numpy as np
import torch

from equistore import TensorMap, Labels, TensorBlock


class LEIterator(torch.nn.Module):

    def __init__(self, E_nl, combined_anl, all_species, cg_object, L_max=None):
        super(LEIterator, self).__init__()

        # Initialize with information that comes from nu = 1 and which is useful at every iteration:
        self.E_nl = E_nl
        self.combined_anl = combined_anl
        self.all_species = all_species
        self.cg_object = cg_object
        self.L_max = L_max

        self.nu_plus_one_count = {}
        self.properties_values = {}
        self.selected_features = {}

    def forward(self, LE_nu, LE_1, E_max_nu_plus_one):

        do_gradients = LE_1.block(0).has_gradient("positions")

        l_max = 0
        for idx, block in LE_1.items():
            l_max = max(l_max, idx["lam"])

        lam_max = 0
        for idx, block in LE_nu.items():
            lam_max = max(lam_max, idx["lam"])

        if self.L_max is None:
            L_max = lam_max + l_max
        else:
            L_max = self.L_max

        nu = len(LE_nu.block(0).properties.names)//4  # Infer nu from the length of the history indices.

        blocks = []
        keys = []

        properties_names = (
            [f"{name}" for name in LE_nu.block(0).properties.names]
            + [f"{name[:-1]}{nu+1}" for name in LE_1.block(0).properties.names]
        )

        for a_i in self.all_species:
            print(f"Combining center species = {a_i}")

            if nu not in self.nu_plus_one_count:

                nu_plus_one_count = {}
                selected_features = {}
                properties_values = {}
                
                for L in range(L_max+1):
                    properties_values[L] = []

                for L in range(L_max+1):
                    nu_plus_one_count[L] = 0

                for lam in range(lam_max+1):
                    block_nu = LE_nu.block(lam=lam, a_i=a_i)
                    a_nu = block_nu.properties["a"+str(nu)]
                    n_nu = block_nu.properties["n"+str(nu)]
                    l_nu = block_nu.properties["l"+str(nu)]

                    for l in range(l_max+1):
                        block_1 = LE_1.block(lam=l, a_i=a_i)
                        a_nu_plus_1 = block_1.properties["a1"]
                        n_nu_plus_1 = block_1.properties["n1"]

                        for L in range(np.abs(lam-l), min(lam+l+1, L_max)):
                            selected_features[(lam, l, L)] = []

                        for q_nu in range(block_nu.values.shape[-1]):
                            for q_1 in range(block_1.values.shape[-1]):
                                # Check lexicographic ordering:
                                if self.combined_anl[(a_nu[q_nu], n_nu[q_nu], l_nu[q_nu])] > self.combined_anl[(a_nu_plus_1[q_1], n_nu_plus_1[q_1], l)]: continue
                                # Check LE eigenvalue:
                                hypothetical_eigenvalue = 0.0
                                for nu in range(1, nu+1):
                                    hypothetical_eigenvalue += self.E_nl[block_nu.properties["n"+str(nu)][q_nu], block_nu.properties["l"+str(nu)][q_nu]]
                                hypothetical_eigenvalue += self.E_nl[n_nu_plus_1[q_1], l]
                                if hypothetical_eigenvalue > E_max_nu_plus_one: continue 

                                properties_list = [[block_nu.properties[name][q_nu] for name in block_nu.properties.names] + [block_1.properties[name][q_1] for name in block_1.properties.names[:-1]] + [lam]]
                                
                                for L in range(np.abs(lam-l), min(lam+l+1, L_max)):
                                    # Avoid calculating trivial nu = 2 zeros:
                                    if nu == 1 and lam == l and L % 2 == 1: continue
                                    nu_plus_one_count[L] += 1
                                    properties_values[L].append(properties_list)
                                    selected_features[(lam, l, L)].append([q_nu, q_1])

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

            block_1 = LE_1.block(lam=0, a_i=a_i)
            data = {}
            if do_gradients: gradient_data = {}
            for L in range(L_max+1):
                data[L] = torch.zeros((len(block_1.samples), 2*L+1, nu_plus_one_count[L]), device=block_1.values.device)
                if do_gradients: gradient_data[L] = torch.zeros((len(block_1.gradient("positions").samples), 3, 2*L+1, nu_plus_one_count[L]), device=block_1.values.device)

            for L in range(L_max+1):
                nu_plus_one_count[L] = 0

            for lam in range(lam_max+1):
                block_nu = LE_nu.block(lam=lam, a_i=a_i)

                for l in range(l_max+1):
                    block_1 = LE_1.block(lam=l, a_i=a_i)

                    for L in range(np.abs(lam-l), min(lam+l+1, L_max)):
                        if (lam, l, L) not in selected_features: continue  # No features are selected.
                        
                        nu_plus_one_values, nu_plus_one_derivatives = self.cg_object.combine(block_nu, block_1, L, selected_features[(lam, l, L)])
                        data[L][:, :, nu_plus_one_count[L]:nu_plus_one_count[L]+selected_features[(lam, l, L)].shape[0]] = nu_plus_one_values

                        if do_gradients:
                            gradient_data[L][:, :, :, nu_plus_one_count[L]:nu_plus_one_count[L]+selected_features[(lam, l, L)].shape[0]] = nu_plus_one_derivatives

                        nu_plus_one_count[L] += selected_features[(lam, l, L)].shape[0]
                        

            for L in range(L_max+1):
                if len(properties_values[L]) == 0:
                    # print(f"L = {L} completely discarded")
                    continue
                block = TensorBlock(
                    values=data[L],
                    samples=block_1.samples,
                    components=[Labels(
                        names=("mu",),
                        values=np.asarray(range(-L, L+1), dtype=np.int32).reshape(2*L+1, 1),
                    )],
                    properties=Labels(
                        names=properties_names,
                        values=np.asarray(np.vstack(properties_values[L]), dtype=np.int32),
                    ),
                )
                if do_gradients: block.add_gradient(
                    parameter="positions",
                    gradient=TensorBlock(
                        values = gradient_data[L], 
                        samples = block_1.gradient("positions").samples, 
                        components = [ 
                            block_1.gradient("positions").components[0],
                            Labels(
                                names=("mu",),
                                values=np.asarray(range(-L, L+1), dtype=np.int32).reshape(2*L+1, 1),
                            ), 
                        ],
                        properties = block.properties
                    )
                )
                blocks.append(block)
                keys.append([a_i, L])

        LE_nu_plus_one = TensorMap(
            keys = Labels(
                names = ("a_i", "lam"),
                values = np.array(keys).reshape((-1, 2)),
            ), 
            blocks = blocks)

        return LE_nu_plus_one

