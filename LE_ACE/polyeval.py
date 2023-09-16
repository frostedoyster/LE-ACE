import torch


def polyeval(nu1_basis, indices, multipliers, species_indices):

    polynomial_order = indices.shape[1]
    indices = indices.to(torch.long)

    product = nu1_basis.index_select(1, indices[:, 0])
    for monomial_index in range(1, polynomial_order):
        product *= nu1_basis.index_select(1, indices[:, monomial_index])
    atomic_energies = torch.sum(product*multipliers.index_select(0, species_indices), dim=1)

    return atomic_energies
