import torch
import numpy as np

def get_LE_regularization(properties, E_nl):

    eigenvalues = []
    nu = len(properties.names)//4
    for i in range(len(properties["n1"])):
        eigenvalue = 0.0
        for iota in range(1, nu+1):
            eigenvalue += E_nl[properties["n"+str(iota)][i], properties["l"+str(iota)][i]]
        eigenvalues.append(eigenvalue) # **nu)

    return torch.tensor(eigenvalues)
