import torch


class ACECalculator(torch.nn.Module):

    def __init__(self, l_max, combine_indices, multiplicities, generalized_cgs):
        super().__init__()

        self.combine_indices = combine_indices
        self.multiplicities = multiplicities
        self.l_max = l_max
        self.nu_max = len(combine_indices) - 1
        self.generalized_cgs = generalized_cgs
        self.final_state = [-1] + [self.l_max] * self.nu_max

    def forward(self, LE_1):
        # LE_1 is in format (l) -> [b, m, i]
        B_basis = {}
        for (L_nu, sigma), generalized_cgs_L_sigma in self.generalized_cgs.items():
            B_basis[(L_nu, sigma)] = [{} for nu in range(self.nu_max+1)]  # FIXME: NOT WORKING YET FOR OTHER CHOICES OF LAMBDA, SIGMA
        L_nu = 0
        sigma = 1

        current_location = [-1, -1] + [self.l_max] * (self.nu_max-1)
        current_A_basis = [torch.empty((0,))] * (self.nu_max+1)

        while current_location != self.final_state:

            if current_location[self.nu_max] == self.l_max:
                # Find 
                where_first_l_max = self.nu_max
                while current_location[where_first_l_max] == self.l_max:
                    where_first_l_max -= 1
                where_first_l_max += 1
                current_location[where_first_l_max-1] += 1
                for nu in range(where_first_l_max, self.nu_max+1):
                    assert current_location[nu] == self.l_max
                    current_location[nu] = 0
                for nu in range(where_first_l_max-1, self.nu_max+1):
                    if nu == 1:
                        l_tuple_nu = (current_location[1],)
                        current_A_basis[nu] = LE_1[l_tuple_nu]
                    else:
                        l_tuple_nu = tuple(current_location[1:nu+1])
                        if l_tuple_nu in self.combine_indices[nu].keys():
                            combine_indices_l_tuple_nu = self.combine_indices[nu][l_tuple_nu]
                            indices_nu_minus_one_l = combine_indices_l_tuple_nu[0]
                            indices_1_l = combine_indices_l_tuple_nu[1]
                            current_A_basis[nu] = current_A_basis[nu-1].index_select(0, indices_nu_minus_one_l).unsqueeze(2) * LE_1[(l_tuple_nu[-1],)].index_select(0, indices_1_l).unsqueeze(1)
                            current_A_basis[nu] = current_A_basis[nu].reshape(current_A_basis[nu].shape[0], -1, current_A_basis[nu].shape[3])
                            if l_tuple_nu in self.generalized_cgs[(L_nu, sigma)][nu].keys():
                                n_i = current_A_basis[nu].shape[2]
                                B_basis[(L_nu, sigma)][nu][l_tuple_nu] = self.generalized_cgs[(L_nu, sigma)][nu][l_tuple_nu] @ (current_A_basis[nu]*self.multiplicities[nu][l_tuple_nu].unsqueeze(1).unsqueeze(2)).swapaxes(0, 1).reshape(current_A_basis[nu].shape[1], -1)
                                B_basis[(L_nu, sigma)][nu][l_tuple_nu] = B_basis[(L_nu, sigma)][nu][l_tuple_nu].reshape(2*L_nu+1, -1, n_i)  # [M, L_tuple and b, i]
            
            else:
                nu = self.nu_max
                current_location[nu] += 1
                l_tuple_nu = tuple(current_location[1:nu+1])
                if l_tuple_nu in self.combine_indices[nu]:
                    combine_indices_l_tuple_nu = self.combine_indices[nu][l_tuple_nu]
                    indices_nu_minus_one_l = combine_indices_l_tuple_nu[0]
                    indices_1_l = combine_indices_l_tuple_nu[1]
                    current_A_basis[nu] = current_A_basis[nu-1].index_select(0, indices_nu_minus_one_l).unsqueeze(2) * LE_1[(l_tuple_nu[-1],)].index_select(0, indices_1_l).unsqueeze(1)
                    current_A_basis[nu] = current_A_basis[nu].reshape(current_A_basis[nu].shape[0], -1, current_A_basis[nu].shape[3])
                    if l_tuple_nu in self.generalized_cgs[(L_nu, sigma)][nu].keys():
                        n_i = current_A_basis[nu].shape[2]
                        B_basis[(L_nu, sigma)][nu][l_tuple_nu] = self.generalized_cgs[(L_nu, sigma)][nu][l_tuple_nu] @ (current_A_basis[nu]*self.multiplicities[nu][l_tuple_nu].unsqueeze(1).unsqueeze(2)).swapaxes(0, 1).reshape(current_A_basis[nu].shape[1], -1)
                        B_basis[(L_nu, sigma)][nu][l_tuple_nu] = B_basis[(L_nu, sigma)][nu][l_tuple_nu].reshape(2*L_nu+1, -1, n_i)  # [M, L_tuple and b, i]

        return B_basis

    def compute_with_gradients(self, LE_1_values, LE_1_gradients, gradient_metadata):
        # LE_1_values is in format (l) -> [b, m, i]
        # LE_1_gradients is in format (l) -> [b, m, ij, alpha]

        B_basis = {}
        B_basis_grad = {}
        for (L_nu, sigma), generalized_cgs_L_sigma in self.generalized_cgs.items():
            B_basis[(L_nu, sigma)] = [{} for nu in range(self.nu_max+1)]  # FIXME: NOT WORKING YET FOR OTHER CHOICES OF LAMBDA, SIGMA
            B_basis_grad[(L_nu, sigma)] = [{} for nu in range(self.nu_max+1)]  # FIXME: NOT WORKING YET FOR OTHER CHOICES OF LAMBDA, SIGMA
        L_nu = 0
        sigma = 1

        current_location = [-1, -1] + [self.l_max] * (self.nu_max-1)
        current_A_basis = [torch.empty((0,))] * (self.nu_max+1)
        current_A_basis_grad = [torch.empty((0,))] * (self.nu_max+1)

        while current_location != self.final_state:

            if current_location[self.nu_max] == self.l_max:
                # Find 
                where_first_l_max = self.nu_max
                while current_location[where_first_l_max] == self.l_max:
                    where_first_l_max -= 1
                where_first_l_max += 1
                current_location[where_first_l_max-1] += 1
                for nu in range(where_first_l_max, self.nu_max+1):
                    assert current_location[nu] == self.l_max
                    current_location[nu] = 0
                for nu in range(where_first_l_max-1, self.nu_max+1):
                    if nu == 1:
                        l_tuple_nu = (current_location[1],)
                        current_A_basis[nu] = LE_1_values[l_tuple_nu]
                        current_A_basis_grad[nu] = LE_1_gradients[l_tuple_nu]
                    else:
                        l_tuple_nu = tuple(current_location[1:nu+1])
                        if l_tuple_nu in self.combine_indices[nu].keys():
                            combine_indices_l_tuple_nu = self.combine_indices[nu][l_tuple_nu]
                            indices_nu_minus_one_l = combine_indices_l_tuple_nu[0]
                            indices_1_l = combine_indices_l_tuple_nu[1]
                            current_A_basis[nu] = current_A_basis[nu-1].index_select(0, indices_nu_minus_one_l).unsqueeze(2) * LE_1_values[(l_tuple_nu[-1],)].index_select(0, indices_1_l).unsqueeze(1)
                            current_A_basis_grad[nu] = (
                                current_A_basis[nu-1].index_select(0, indices_nu_minus_one_l).index_select(2, gradient_metadata).unsqueeze(2).unsqueeze(4)*LE_1_gradients[(l_tuple_nu[-1],)].index_select(0, indices_1_l).unsqueeze(1)
                                +
                                current_A_basis_grad[nu-1].index_select(0, indices_nu_minus_one_l).unsqueeze(2)*LE_1_values[(l_tuple_nu[-1],)].index_select(0, indices_1_l).index_select(2, gradient_metadata).unsqueeze(1).unsqueeze(4)
                            )
                            current_A_basis[nu] = current_A_basis[nu].reshape(current_A_basis[nu].shape[0], -1, current_A_basis[nu].shape[3])
                            current_A_basis_grad[nu] = current_A_basis_grad[nu].reshape(current_A_basis_grad[nu].shape[0], -1, current_A_basis_grad[nu].shape[3], 3)
                            if l_tuple_nu in self.generalized_cgs[(L_nu, sigma)][nu].keys():  # this if shouldn't be necessary
                                n_i = current_A_basis[nu].shape[2]
                                n_ij = current_A_basis_grad[nu].shape[2]
                                B_basis[(L_nu, sigma)][nu][l_tuple_nu] = self.generalized_cgs[(L_nu, sigma)][nu][l_tuple_nu] @ (current_A_basis[nu]*self.multiplicities[nu][l_tuple_nu].unsqueeze(1).unsqueeze(2)).swapaxes(0, 1).reshape(current_A_basis[nu].shape[1], -1)
                                B_basis[(L_nu, sigma)][nu][l_tuple_nu] = B_basis[(L_nu, sigma)][nu][l_tuple_nu].reshape(2*L_nu+1, -1, n_i)  # [M, L_tuple and b, i]
                                B_basis_grad[(L_nu, sigma)][nu][l_tuple_nu] = self.generalized_cgs[(L_nu, sigma)][nu][l_tuple_nu] @ (current_A_basis_grad[nu]*self.multiplicities[nu][l_tuple_nu].unsqueeze(1).unsqueeze(2).unsqueeze(3)).swapaxes(0, 1).reshape(current_A_basis_grad[nu].shape[1], -1)
                                B_basis_grad[(L_nu, sigma)][nu][l_tuple_nu] = B_basis_grad[(L_nu, sigma)][nu][l_tuple_nu].reshape(2*L_nu+1, -1, n_ij, 3)  # [M, L_tuple and b, ij, alpha]
            else:
                nu = self.nu_max
                current_location[nu] += 1
                l_tuple_nu = tuple(current_location[1:nu+1])
                if l_tuple_nu in self.combine_indices[nu].keys():
                    combine_indices_l_tuple_nu = self.combine_indices[nu][l_tuple_nu]
                    indices_nu_minus_one_l = combine_indices_l_tuple_nu[0]
                    indices_1_l = combine_indices_l_tuple_nu[1]
                    current_A_basis[nu] = current_A_basis[nu-1].index_select(0, indices_nu_minus_one_l).unsqueeze(2) * LE_1_values[(l_tuple_nu[-1],)].index_select(0, indices_1_l).unsqueeze(1)
                    current_A_basis_grad[nu] = (
                        current_A_basis[nu-1].index_select(0, indices_nu_minus_one_l).index_select(2, gradient_metadata).unsqueeze(2).unsqueeze(4)*LE_1_gradients[(l_tuple_nu[-1],)].index_select(0, indices_1_l).unsqueeze(1)
                        +
                        current_A_basis_grad[nu-1].index_select(0, indices_nu_minus_one_l).unsqueeze(2)*LE_1_values[(l_tuple_nu[-1],)].index_select(0, indices_1_l).index_select(2, gradient_metadata).unsqueeze(1).unsqueeze(4)
                    )
                    current_A_basis[nu] = current_A_basis[nu].reshape(current_A_basis[nu].shape[0], -1, current_A_basis[nu].shape[3])
                    current_A_basis_grad[nu] = current_A_basis_grad[nu].reshape(current_A_basis_grad[nu].shape[0], -1, current_A_basis_grad[nu].shape[3], 3)
                    if l_tuple_nu in self.generalized_cgs[(L_nu, sigma)][nu].keys():  # this if shouldn't be necessary
                        n_i = current_A_basis[nu].shape[2]
                        n_ij = current_A_basis_grad[nu].shape[2]
                        B_basis[(L_nu, sigma)][nu][l_tuple_nu] = self.generalized_cgs[(L_nu, sigma)][nu][l_tuple_nu] @ (current_A_basis[nu]*self.multiplicities[nu][l_tuple_nu].unsqueeze(1).unsqueeze(2)).swapaxes(0, 1).reshape(current_A_basis[nu].shape[1], -1)
                        B_basis[(L_nu, sigma)][nu][l_tuple_nu] = B_basis[(L_nu, sigma)][nu][l_tuple_nu].reshape(2*L_nu+1, -1, n_i)  # [M, L_tuple and b, i]
                        B_basis_grad[(L_nu, sigma)][nu][l_tuple_nu] = self.generalized_cgs[(L_nu, sigma)][nu][l_tuple_nu] @ (current_A_basis_grad[nu]*self.multiplicities[nu][l_tuple_nu].unsqueeze(1).unsqueeze(2).unsqueeze(3)).swapaxes(0, 1).reshape(current_A_basis_grad[nu].shape[1], -1)
                        B_basis_grad[(L_nu, sigma)][nu][l_tuple_nu] = B_basis_grad[(L_nu, sigma)][nu][l_tuple_nu].reshape(2*L_nu+1, -1, n_ij, 3)  # [M, L_tuple and b, ij, alpha]

        return B_basis, B_basis_grad
