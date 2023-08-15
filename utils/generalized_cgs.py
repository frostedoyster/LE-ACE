import numpy as np
import torch
import wigners


def get_generalized_cgs(l_tuples, requested_L_sigma, nu_max, device):
    # At the moment, we are not removing anything, not even the trivial zeros.
    # It should be noted that these are not zeros of the generalized CGs, but they arise due to the combination
    # of constraints in the generalized CGs and in the spherical harmonics.

    cg_calculator = ClebschGordanReal(device)

    generalized_cgs = {L_sigma: [{}, {}] for L_sigma in requested_L_sigma}
    for L_nu, sigma in requested_L_sigma:
        generalized_cgs_L_sigma = generalized_cgs[(L_nu, sigma)]
        for nu in range(2, nu_max+1):
            l_tuples_nu = l_tuples[nu]
            generalized_cgs_L_sigma.append({})

            for l_tuple in l_tuples_nu:

                # check parity:
                if (-1)**(sum(l_tuple)+L_nu) != sigma: continue
                generalized_cgs_iota_minus_one = {(l_tuple[0],): torch.eye(2*l_tuple[0]+1, dtype=torch.get_default_dtype(), device=device)}

                for iota in range(2, nu+1):
                    generalized_cgs_iota = {}
                    for L_tuple, generalized_cg_matrix_iota_minus_one in generalized_cgs_iota_minus_one.items():
                        L_iota_minus_one = L_tuple[-1]
                        l_iota = l_tuple[iota-1]

                        if iota != nu:
                            for L_iota in range(abs(L_iota_minus_one-l_iota), L_iota_minus_one+l_iota+1):
                                generalized_cgs_iota[L_tuple+(L_iota,)] = (
                                    generalized_cg_matrix_iota_minus_one @ cg_calculator.get((L_iota_minus_one, l_iota, L_iota)).reshape(2*L_iota_minus_one+1, -1)
                                ).reshape([size for size in generalized_cg_matrix_iota_minus_one.shape[:-1]]+[2*l_iota+1]+[2*L_iota+1])
                        else:
                            L_iota = L_nu
                            if L_iota < abs(L_iota_minus_one-l_iota) or L_iota > L_iota_minus_one+l_iota: continue
                            generalized_cgs_iota[L_tuple+(L_iota,)] = (
                                generalized_cg_matrix_iota_minus_one @ cg_calculator.get((L_iota_minus_one, l_iota, L_iota)).reshape(2*L_iota_minus_one+1, -1)
                            ).reshape([size for size in generalized_cg_matrix_iota_minus_one.shape[:-1]]+[2*l_iota+1]+[2*L_iota+1]).reshape(-1, 2*L_iota+1).T
                        
                    generalized_cgs_iota_minus_one = generalized_cgs_iota
                
                if len(generalized_cgs_iota) > 0: generalized_cgs[(L_nu, sigma)][nu][(l_tuple)] = generalized_cgs_iota

    # Make more compace by stacking over the L_tuple dimension:
    generalized_cgs_compressed = {L_sigma: [{}, {}] for L_sigma in requested_L_sigma}
    for L_sigma in requested_L_sigma:
        generalized_cgs_compressed_L_sigma = generalized_cgs_compressed[L_sigma]
        for nu in range(2, nu_max+1):
            l_tuples_nu = l_tuples[nu]
            generalized_cgs_compressed_L_sigma.append({})
            for l_tuple in generalized_cgs[L_sigma][nu].keys():
                generalized_cg_matrices_l_tuple = list(generalized_cgs[L_sigma][nu][l_tuple].values())
                generalized_cgs_compressed_L_sigma[nu][l_tuple] = torch.stack(generalized_cg_matrices_l_tuple, dim=1)
                generalized_cgs_compressed_L_sigma[nu][l_tuple] = generalized_cgs_compressed_L_sigma[nu][l_tuple].reshape(-1, generalized_cgs_compressed_L_sigma[nu][l_tuple].shape[2]).to_sparse_csr()

    return generalized_cgs_compressed 


class ClebschGordanReal:

    def __init__(self, device):
        self._cgs = {}
        self.device = device

    def _add(self, l1, l2, L):
        # print(f"Adding new CGs with l1={l1}, l2={l2}, L={L}")

        if self._cgs is None: 
            raise ValueError("Trying to add CGs when not initialized... exiting")

        if (l1, l2, L) in self._cgs: 
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

        # Zero any possible (and very rare) near-zero elements
        where_almost_zero = np.where(np.logical_and(np.abs(rcg) > 0, np.abs(rcg) < 1e-14))
        if len(where_almost_zero[0] != 0):
            print("INFO: Found almost-zero CG!")
        for i0, i1, i2 in zip(where_almost_zero[0], where_almost_zero[1], where_almost_zero[2]):
            rcg[i0, i1, i2] = 0.0

        self._cgs[(l1, l2, L)] = torch.tensor(rcg).type(torch.get_default_dtype()).to(self.device)

    def get(self, key):
        if key in self._cgs:
            return self._cgs[key]
        else:
            self._add(key[0], key[1], key[2])
            return self._cgs[key]


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

if __name__ == "__main__":
    l_tuples = [
        0,
        1,
        [(0, 0)],
        [(3, 4, 7), (2, 2, 2)]
    ]
    cgs = get_generalized_cgs(l_tuples, [(0, 1)], 3, "cpu")
    print(cgs)
