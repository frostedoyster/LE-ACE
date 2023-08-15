import torch


class ACESymmetrizer(torch.nn.Module):

    def __init__(self, generalized_cgs):
        super().__init__()

        self.generalized_cgs = generalized_cgs
        self.nu_max = len(list(generalized_cgs.values())[0]) - 1

    def forward(self, A_basis):
        # A-basis coming in as (l_tuple) -> [b, m, i]

        B_basis = {}
        for (L_nu, sigma), generalized_cgs_L_sigma in self.generalized_cgs.items():
            B_basis[(L_nu, sigma)] = [{}, {}]
            for nu in range(2, self.nu_max+1):
                B_basis[(L_nu, sigma)].append({})
                for l_tuple, generalized_cgs_L_sigma_l_tuple in generalized_cgs_L_sigma[nu].items():
                    n_i = A_basis[nu][l_tuple].shape[2]
                    B_basis[(L_nu, sigma)][nu][l_tuple] = generalized_cgs_L_sigma_l_tuple @ A_basis[nu][l_tuple].swapaxes(0, 1).reshape(A_basis[nu][l_tuple].shape[1], -1)
                    B_basis[(L_nu, sigma)][nu][l_tuple] = B_basis[(L_nu, sigma)][nu][l_tuple].reshape(2*L_nu+1, -1, n_i)  # [M, L_tuple and b, i]

        return B_basis


    def compute_with_gradients(self, A_basis, A_basis_grad):
        # A-basis coming in as (l_tuple) -> [b, m, i]
        # A-basis gradients coming in as (l_tuple) -> [b, m, ij, alpha]

        B_basis = {}
        B_basis_grad = {}
        for (L_nu, sigma), generalized_cgs_L_sigma in self.generalized_cgs.items():
            B_basis[(L_nu, sigma)] = [{}, {}]
            B_basis_grad[(L_nu, sigma)] = [{}, {}]
            for nu in range(2, self.nu_max+1):
                B_basis[(L_nu, sigma)].append({})
                B_basis_grad[(L_nu, sigma)].append({})
                for l_tuple, generalized_cgs_L_sigma_l_tuple in generalized_cgs_L_sigma[nu].items():
                    n_i = A_basis[nu][l_tuple].shape[2]
                    n_ij = A_basis_grad[nu][l_tuple].shape[2]
                    B_basis[(L_nu, sigma)][nu][l_tuple] = generalized_cgs_L_sigma_l_tuple @ A_basis[nu][l_tuple].swapaxes(0, 1).reshape(A_basis[nu][l_tuple].shape[1], -1)
                    B_basis[(L_nu, sigma)][nu][l_tuple] = B_basis[(L_nu, sigma)][nu][l_tuple].reshape(2*L_nu+1, -1, n_i)  # [M, L_tuple and b, i]
                    B_basis_grad[(L_nu, sigma)][nu][l_tuple] = generalized_cgs_L_sigma_l_tuple @ A_basis_grad[nu][l_tuple].swapaxes(0, 1).reshape(A_basis_grad[nu][l_tuple].shape[1], -1)
                    B_basis_grad[(L_nu, sigma)][nu][l_tuple] = B_basis_grad[(L_nu, sigma)][nu][l_tuple].reshape(2*L_nu+1, -1, n_ij, 3)  # [M, L_tuple and b, ij, alpha]

        return B_basis, B_basis_grad
