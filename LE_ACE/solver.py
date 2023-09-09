import torch


class Solver(torch.nn.Module):
    def __init__(self, n_feat, LE_reg_base, alpha_start, beta_start, nu_max):
        super().__init__()

        self.n_feat = n_feat
        self.nu_max = nu_max
        self.LE_reg_base = LE_reg_base
        self.alpha = torch.nn.Parameter(torch.tensor(alpha_start))
        self.beta = torch.nn.Parameter(torch.tensor(beta_start))

    def forward(self, symm, vec):
        
        LE_reg = [tensor.clone() for tensor in self.LE_reg_base]
        for nu in range(self.nu_max+1):
            LE_reg[nu] *= torch.exp(self.beta*nu)
        LE_reg = torch.concatenate(LE_reg)
        assert LE_reg.shape[0] == symm.shape[0]

        LE_reg = (10**self.alpha)*LE_reg

        c = torch.linalg.solve(symm + torch.diag(LE_reg), vec)
        return c
