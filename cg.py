import numpy as np
import wigners


class ClebschGordanReal:
    def __init__(self, l1_max, l2_max, L_max=None):
        self._l1_max = l1_max
        self._l2_max = l2_max
        if L_max == None:
            self._L_max = self._l1_max + self._l2_max
        else:
            self._L_max = L_max
        self._cg = {}

        maxx = max(self._L_max, max(self._l1_max, self._l2_max))

        # real-to-complex and complex-to-real transformations as matrices
        r2c = {}
        c2r = {}
        for l in range(0, maxx + 1):
            r2c[l] = _real2complex(l)
            c2r[l] = np.conjugate(r2c[l]).T

        for l1 in range(self._l1_max + 1):
            for l2 in range(self._l2_max + 1):
                for L in range(
                    max(l1, l2) - min(l1, l2), min(l1+l2, self._L_max) + 1
                ):
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

                    new_cg = []                    
                    for M in range(2 * L + 1):
                        cg_nonzero = np.where(np.abs(rcg[:,:,M])>1e-15)                        
                        cg_M = np.zeros( len(cg_nonzero[0]), 
                                        dtype=[('m1','>i4'), ('m2','>i4'), ( 'cg', '>f8')] )
                        cg_M["m1"] = cg_nonzero[0]
                        cg_M["m2"] = cg_nonzero[1]
                        cg_M["cg"] = rcg[cg_nonzero[0], cg_nonzero[1], M]
                        new_cg.append(cg_M)
                        
                    self._cg[(l1, l2, L)] = new_cg


    def add(self, l1_max, l2_max, L_max=None):

        if self._cg is None: 
            print("Trying to add CGs when not initialized... exiting")
            exit()

        self._l1_max = l1_max
        self._l2_max = l2_max
        if L_max == None:
            self._L_max = self._l1_max + self._l2_max
        else:
            self._L_max = L_max

        maxx = max(self._L_max, max(self._l1_max, self._l2_max))

        # real-to-complex and complex-to-real transformations as matrices
        r2c = {}
        c2r = {}
        for l in range(0, maxx + 1):
            r2c[l] = _real2complex(l)
            c2r[l] = np.conjugate(r2c[l]).T

        for l1 in range(self._l1_max + 1):
            for l2 in range(self._l2_max + 1):
                for L in range(
                    max(l1, l2) - min(l1, l2), min(l1+l2, self._L_max) + 1
                ):
                    if (l1, l2, L) in self._cg: continue

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

                    new_cg = []                    
                    for M in range(2 * L + 1):
                        cg_nonzero = np.where(np.abs(rcg[:,:,M])>1e-15)                        
                        cg_M = np.zeros( len(cg_nonzero[0]), 
                                        dtype=[('m1','>i4'), ('m2','>i4'), ( 'cg', '>f8')] )
                        cg_M["m1"] = cg_nonzero[0]
                        cg_M["m2"] = cg_nonzero[1]
                        cg_M["cg"] = rcg[cg_nonzero[0], cg_nonzero[1], M]
                        new_cg.append(cg_M)
                        
                    self._cg[(l1, l2, L)] = new_cg
                    

    def combine(self, rho1, rho2, L):
        # automatically infer l1 and l2 from the size of the coefficients vectors
        l1 = (rho1.shape[1] - 1) // 2
        l2 = (rho2.shape[1] - 1) // 2
        if L > self._L_max or l1 > self._l1_max or l2 > self._l2_max:
            raise ValueError("Requested CG entry has not been precomputed")

        n_items = rho1.shape[0]
        n_features = rho1.shape[2]
        if rho1.shape[0] != rho2.shape[0] or rho1.shape[2] != rho2.shape[2]:
            raise IndexError("Cannot combine differently-shaped feature blocks")

        rho = np.zeros((n_items, 2 * L + 1, n_features))
        if (l1, l2, L) in self._cg:
            for M in range(2 * L + 1):
                for m1, m2, cg in self._cg[(l1, l2, L)][M]:
                    rho[:, M] += rho1[:, m1, :] * rho2[:, m2, :] * cg

        return rho

    def combine_einsum(self, rho1, rho2, L, combination_string):
        # automatically infer l1 and l2 from the size of the coefficients vectors
        l1 = (rho1.shape[1] - 1) // 2
        l2 = (rho2.shape[1] - 1) // 2
        if L > self._L_max or l1 > self._l1_max or l2 > self._l2_max:
            raise ValueError("Requested CG entry ", (l1, l2, L), " has not been precomputed")

        n_items = rho1.shape[0]
        if rho1.shape[0] != rho2.shape[0]:
            raise IndexError(
                "Cannot combine feature blocks with different number of items"
            )

        # infers the shape of the output using the einsum internals
        features = np.einsum(combination_string, rho1[:, 0, ...], rho2[:, 0, ...]).shape
        rho = np.zeros((n_items, 2 * L + 1) + features[1:])

        if (l1, l2, L) in self._cg:
            for M in range(2 * L + 1):
                for m1, m2, cg in self._cg[(l1, l2, L)][M]:
                    rho[:, M, ...] += np.einsum(
                        combination_string, rho1[:, m1, ...], rho2[:, m2, ...] * cg
                    )

        return rho


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