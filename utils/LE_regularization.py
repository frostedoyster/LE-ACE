import torch
import numpy as np

def get_LE_regularization(properties, E_nl, r_cut_rs, r_cut):

    eigenvalues = []
    nu = len(properties.names)//4
    for i in range(len(properties["n1"])):
        eigenvalue = 0.0
        for iota in range(1, nu+1):
            eigenvalue += E_nl[properties["n"+str(iota)][i], properties["l"+str(iota)][i]]#*np.exp(-1.5*nu)
        if nu == 1: eigenvalue = eigenvalue*(r_cut**2)/(r_cut_rs**2)
        eigenvalues.append(eigenvalue)

    return torch.tensor(eigenvalues)
