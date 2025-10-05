import numpy as np    


def get_LE_cutoff(E_nl, E_max, rs: bool):
    if rs:
        l_max = 0 
        n_max = np.where(E_nl <= E_max)[0][-1] + 1
        n_max_l = [n_max]
        print(f"Radial spectrum: n_max = {n_max}")
    else:
        n_max = np.where(E_nl[:, 0] <= E_max)[0][-1] + 1
        l_max = np.where(E_nl[0, :] <= E_max)[0][-1]
        print(f"Spherical expansion: n_max = {n_max}, l_max = {l_max}")
        n_max_l = []
        for l in range(l_max+1):
            n_max_l.append(np.where(E_nl[:, l] <= E_max)[0][-1] + 1)

    return n_max_l
