import numpy as np
import torch
import wigners
from collections import namedtuple
import sparse_accumulation

SparseCGTensor = namedtuple("SparseCGTensor", "m1 m2 M cg")

class ClebschGordanReal:
    def __init__(self, device, algorithm = "python loops"):
        self._cg = {}
        self.device = device
        self.algorithm = algorithm


    def _add(self, l1, l2, L):
        # print(f"Adding new CGs with l1={l1}, l2={l2}, L={L}")

        if self._cg is None: 
            raise ValueError("Trying to add CGs when not initialized... exiting")

        if (l1, l2, L) in self._cg: 
            raise ValueError("Trying to add CGs that are already present... exiting")

        maxx = max(l1, max(l2, L))

        # real-to-complex and complex-to-real transformations as matrices
        r2c = {}
        c2r = {}
        for l in range(0, maxx + 1):
            r2c[l] = _real2complex(l)
            c2r[l] = np.conjugate(r2c[l]).T

        complex_cg = _complex_clebsch_gordan_matrix(l1, l2, L)

        real_cg = (r2c[l1].T @ complex_cg.reshape(2 * l1 + 1, -1)).reshape(
            complex_cg.shape
        )

        real_cg = real_cg.swapaxes(0, 1)
        real_cg = (r2c[l2].T @ real_cg.reshape(2 * l2 + 1, -1)).reshape(
            real_cg.shape
        )
        real_cg = real_cg.swapaxes(0, 1)

        real_cg = real_cg @ c2r[L].T

        if (l1 + l2 + L) % 2 == 0:
            rcg = np.real(real_cg)
        else:
            rcg = np.imag(real_cg)

        if self.algorithm == "dense":

            self._cg[(l1, l2, L)] = torch.tensor(rcg).type(torch.get_default_dtype()).to(self.device)

        else:

            # Note that this construction already provides "aligned" CG 
            # coefficients ready to be used in the sparse accumulation package:

            m1_array = [] 
            m2_array = []
            M_array = []
            cg_array = []

            for M in range(2 * L + 1):
                cg_nonzero = np.where(np.abs(rcg[:,:,M]) > 1e-15)
                m1_array.append(cg_nonzero[0])
                m2_array.append(cg_nonzero[1])
                M_array.append(np.array([M]*len(cg_nonzero[0])))
                cg_array.append(rcg[cg_nonzero[0], cg_nonzero[1], M])

            m1_array = torch.LongTensor(np.concatenate(m1_array)).to(self.device)
            m2_array = torch.LongTensor(np.concatenate(m2_array)).to(self.device)
            M_array = torch.LongTensor(np.concatenate(M_array)).to(self.device)
            cg_array = torch.tensor(np.concatenate(cg_array)).type(torch.get_default_dtype()).to(self.device)

            new_cg = SparseCGTensor(m1_array, m2_array, M_array, cg_array)
                
            self._cg[(l1, l2, L)] = new_cg


    def combine(self, block_nu, block_1, L, selected_features):

        device = self.device

        lam = (block_nu.values.shape[1] - 1) // 2
        l = (block_1.values.shape[1] - 1) // 2

        if (lam, l, L) not in self._cg:
            self._add(lam, l, L)

        if self.algorithm != "dense":
            sparse_cg_tensor = self._cg[(lam, l, L)]
            mu_array = sparse_cg_tensor.m1
            m_array = sparse_cg_tensor.m2
            M_array = sparse_cg_tensor.M
            cg_array = sparse_cg_tensor.cg
        else:  # dense algorithm
            dense_cg_matrix = self._cg[(lam, l, L)]
            dense_cg_matrix = dense_cg_matrix.reshape(-1, dense_cg_matrix.shape[2])

        if block_nu.has_gradient("positions"):
            gradients_nu = block_nu.gradient("positions")
            samples_for_gradients_nu = torch.tensor(gradients_nu.samples["sample"], dtype=torch.int64)
            gradients_1 = block_1.gradient("positions")
            samples_for_gradients_1 = torch.tensor(gradients_1.samples["sample"], dtype=torch.int64)
            n_selected_features = selected_features.shape[0]

        if self.algorithm == "fast cg":

            if device == "cpu":

                new_values = sparse_accumulation.accumulate_active_dim_middle(
                    block_nu.values[:, :, selected_features[:, 0]].contiguous(),
                    block_1.values[:, :, selected_features[:, 1]].contiguous(), 
                    M_array, 
                    2*L+1, 
                    mu_array, 
                    m_array, 
                    cg_array
                )

                if block_nu.has_gradient("positions"):

                    new_derivatives = (
                        sparse_accumulation.accumulate_active_dim_middle(
                            gradients_nu.values[:, :, :, selected_features[:, 0]].reshape((-1, 2*lam+1, n_selected_features)).contiguous(),
                            block_1.values[samples_for_gradients_nu][:, :, selected_features[:, 1]].unsqueeze(dim=1)[:, [0, 0, 0], :, :].reshape((-1, 2*l+1, n_selected_features)).contiguous(),
                            M_array, 
                            2*L+1, 
                            mu_array, 
                            m_array, 
                            cg_array
                        ) + sparse_accumulation.accumulate_active_dim_middle(
                            block_nu.values[samples_for_gradients_1][:, :, selected_features[:, 0]].unsqueeze(dim=1)[:, [0, 0, 0], :, :].reshape((-1, 2*lam+1, n_selected_features)).contiguous(),
                            gradients_1.values[:, :, :, selected_features[:, 1]].reshape((-1, 2*l+1, n_selected_features)).contiguous(),
                            M_array, 
                            2*L+1, 
                            mu_array, 
                            m_array, 
                            cg_array
                        )
                    ).reshape((-1, 3, 2*L+1, n_selected_features))
                else:
                    new_derivatives = None

            else:  # GPU device

                new_values = sparse_accumulation.accumulate(
                    block_nu.values[:, :, selected_features[:, 0]].swapaxes(1, 2).contiguous(),
                    block_1.values[:, :, selected_features[:, 1]].swapaxes(1, 2).contiguous(), 
                    M_array, 
                    2*L+1, 
                    mu_array, 
                    m_array, 
                    cg_array
                ).swapaxes(1, 2)

                if block_nu.has_gradient("positions"):

                    new_derivatives = (
                        sparse_accumulation.accumulate(
                            gradients_nu.values[:, :, :, selected_features[:, 0]].reshape((-1, 2*lam+1, n_selected_features)).swapaxes(1, 2).contiguous(),
                            block_1.values[samples_for_gradients_nu][:, :, selected_features[:, 1]].unsqueeze(dim=1)[:, [0, 0, 0], :, :].reshape((-1, 2*l+1, n_selected_features)).swapaxes(1, 2).contiguous(),
                            M_array, 
                            2*L+1, 
                            mu_array, 
                            m_array, 
                            cg_array
                        ) + sparse_accumulation.accumulate(
                            block_nu.values[samples_for_gradients_1][:, :, selected_features[:, 0]].unsqueeze(dim=1)[:, [0, 0, 0], :, :].reshape((-1, 2*lam+1, n_selected_features)).swapaxes(1, 2).contiguous(),
                            gradients_1.values[:, :, :, selected_features[:, 1]].reshape((-1, 2*l+1, n_selected_features)).swapaxes(1, 2).contiguous(),
                            M_array, 
                            2*L+1, 
                            mu_array, 
                            m_array, 
                            cg_array
                        )
                    ).swapaxes(1, 2).reshape((-1, 3, 2*L+1, n_selected_features))
                else:
                    new_derivatives = None

        elif self.algorithm == "dense":

            new_values = (
                (block_nu.values[:, :, selected_features[:, 0]].swapaxes(1, 2))[:, :, :, None] * 
                (block_1.values[:, :, selected_features[:, 1]].swapaxes(1, 2))[:, :, None, :]
            )
            new_values = new_values.reshape((new_values.shape[0], new_values.shape[1], -1))
            new_values = new_values @ dense_cg_matrix
            new_values = new_values.swapaxes(1, 2)

            if block_nu.has_gradient("positions"):

                new_derivatives_1 = (
                    (gradients_nu.values[:, :, :, selected_features[:, 0]].reshape((-1, 2*lam+1, n_selected_features)).swapaxes(1, 2))[:, :, :, None] *
                    (block_1.values[samples_for_gradients_nu][:, :, selected_features[:, 1]].unsqueeze(dim=1)[:, [0, 0, 0], :, :].reshape((-1, 2*l+1, n_selected_features)).swapaxes(1, 2))[:, :, None, :]
                )
                new_derivatives_2 = (
                    (block_nu.values[samples_for_gradients_1][:, :, selected_features[:, 0]].unsqueeze(dim=1)[:, [0, 0, 0], :, :].reshape((-1, 2*lam+1, n_selected_features)).swapaxes(1, 2))[:, :, :, None] *
                    (gradients_1.values[:, :, :, selected_features[:, 1]].reshape((-1, 2*l+1, n_selected_features)).swapaxes(1, 2))[:, :, None, :]
                )
                new_derivatives = new_derivatives_1 + new_derivatives_2
                new_derivatives = new_derivatives.reshape((new_derivatives.shape[0], new_derivatives.shape[1], -1))
                new_derivatives = new_derivatives @ dense_cg_matrix
                new_derivatives = new_derivatives.swapaxes(1, 2).reshape((-1, 3, 2*L+1, n_selected_features))

            else:
                new_derivatives = None


        else:  # Python loop algorithm

            new_values = torch.zeros((block_nu.values.shape[0], 2*L+1, selected_features.shape[0]), device=device)
            if block_nu.has_gradient("positions"): 
                new_derivatives = torch.zeros((gradients_nu.values.shape[0], 3, 2*L+1, selected_features.shape[0]), device=device)
            else:
                new_derivatives = None

            for mu, m, M, cg_coefficient in zip(sparse_cg_tensor.m1, sparse_cg_tensor.m2, sparse_cg_tensor.M, sparse_cg_tensor.cg):
                new_values[:, M, :] += cg_coefficient * block_nu.values[:, mu, selected_features[:, 0]] * block_1.values[:, m, selected_features[:, 1]]
                if block_nu.has_gradient("positions"): 
                    new_derivatives[:, :, M, :] += cg_coefficient * (gradients_nu.values[:, :, mu, selected_features[:, 0]] * block_1.values[samples_for_gradients_nu][:, m, selected_features[:, 1]].unsqueeze(dim=1) + block_nu.values[samples_for_gradients_1][:, mu, selected_features[:, 0]].unsqueeze(dim=1) * gradients_1.values[:, :, m, selected_features[:, 1]])  # exploiting broadcasting rules

        return new_values, new_derivatives


def _real2complex(L):
    """
    Computes a matrix that can be used to convert from real to complex-valued
    spherical harmonics(coefficients) of order L.

    It's meant to be applied to the left, ``real2complex @ [-L..L]``.
    """
    result = np.zeros((2 * L + 1, 2 * L + 1), dtype=np.complex128)

    I_SQRT_2 = 1.0 / np.sqrt(2)

    for m in range(-L, L + 1):
        if m < 0:
            result[L - m, L + m] = I_SQRT_2 * 1j * (-1) ** m
            result[L + m, L + m] = -I_SQRT_2 * 1j

        if m == 0:
            result[L, L] = 1.0

        if m > 0:
            result[L + m, L + m] = I_SQRT_2 * (-1) ** m
            result[L - m, L + m] = I_SQRT_2

    return result


def _complex_clebsch_gordan_matrix(l1, l2, L):
    if np.abs(l1 - l2) > L or np.abs(l1 + l2) < L:
        return np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * L + 1), dtype=np.double)
    else:
        return wigners.clebsch_gordan_array(l1, l2, L)
