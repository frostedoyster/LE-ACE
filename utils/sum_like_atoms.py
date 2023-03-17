import numpy as np
import torch
import equistore


def sum_like_atoms(invariants):

    summed_invariants = []
    for nu_minus_one in range(len(invariants)):
        summed_invariants.append(equistore.sum_over_samples(invariants[nu_minus_one], "center"))
    return summed_invariants
