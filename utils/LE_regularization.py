import torch
import numpy as np

def get_LE_regularization(properties, E_nl, r_cut_rs, r_cut, beta):

    eigenvalues = []
    assert (len(properties.names)-1)%4 == 0
    nu = (len(properties.names)-1)//4
    for i in range(len(properties["n1"])):
        if nu == 1:
            eigenvalue = E_nl[properties["n1"][i]]*np.exp(beta*nu)
        else:
            eigenvalue = 0.0
            for iota in range(1, nu+1):
                eigenvalue += E_nl[properties["n"+str(iota)][i], properties["l"+str(iota)][i]]*np.exp(beta*nu)  # reproduce
        if nu == 1: eigenvalue = eigenvalue*(r_cut**2)/(r_cut_rs**2)   # This only makes sense for pure LE and paper versions  # reproduce
        eigenvalues.append(eigenvalue)

    return torch.tensor(eigenvalues)
