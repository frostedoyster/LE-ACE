import numpy as np    


def get_LE_cutoff(E_nl, E_max, rs):
    if rs:
        l_max = 0 
        n_max = np.where(E_nl <= E_max)[0][-1] + 1
        print(f"Radial spectrum: n_max = {n_max}")
    else:
        n_max = np.where(E_nl[:, 0] <= E_max)[0][-1] + 1
        l_max = np.where(E_nl[0, :] <= E_max)[0][-1]
        print(f"Spherical expansion: n_max = {n_max}, l_max = {l_max}")

    return n_max, l_max
