import torch
import math
import numpy as np


def get_le_metadata(all_species, E_max, n_max_l, E_ln, is_trace, device):
    nu_max = len(E_max) - 1
    l_max = len(n_max_l) - 1
    n_elems = len(all_species)

    # generate b, the everything index
    if is_trace:
        a_max = 1  # Pseudo-element channels don't couple to one another
    else:
        a_max = n_elems
    b = {}
    b_counter = 0
    for l in range(l_max+1):
        for a in range(a_max):
            for n in range(n_max_l[l]):
                if is_trace:
                    b[(l, n)] = b_counter
                else:
                    b[(l, a, n)] = b_counter
                b_counter += 1

    # a_max ALL WRONG FOR TRACE. From now on, should rather use the number of different element channels

    combine_indices = [0, 1]
    multiplicities = [0, {(l,): torch.ones((a_max*n_max_l[l],), dtype=torch.long, device=device) for l in range(l_max+1)}]
    LE_energies = [0, {(l,): torch.tile(torch.tensor(E_ln[l], dtype=torch.get_default_dtype(), device=device), (a_max,)) for l in range(l_max+1)}]
    
    b_1 = {}
    for l in range(l_max+1):
        b_1_l = []
        for a in range(a_max):
            for n in range(n_max_l[l]):
                if is_trace:
                    b_1_l.append(b[(l, n)])
                else:
                    b_1_l.append(b[(l, a, n)])
        b_1[(l,)] = torch.tensor(b_1_l, dtype=torch.long, device=device).unsqueeze(1) 

    b_nu_minus_one = b_1
    for nu in range(2, nu_max+1):

        LE_energies_1 = LE_energies[1]
        LE_energies_nu_minus_one = LE_energies[nu-1]
        LE_energies.append({})
        combine_indices.append({})
        multiplicities.append({})
        b_nu = {}

        for l_tuple_nu_minus_one, LE_energies_nu_minus_one_l in LE_energies_nu_minus_one.items():
            b_nu_minus_one_l_tuple = b_nu_minus_one[l_tuple_nu_minus_one]

            for (l,), LE_energies_1_l in LE_energies_1.items():
                if l_tuple_nu_minus_one[-1] > l: continue
                b_1_l = b_1[(l,)]
                l_tuple_nu = l_tuple_nu_minus_one + (l,)
                # print(l_tuple_nu)
                LE_energies[nu][l_tuple_nu] = []
                b_nu[l_tuple_nu] = []
                combine_indices_l_tuple_nu_minus_one = []
                combine_indices_l_tuple_1 = []

                for p_nu_minus_one, LE_energy_nu_minus_one in enumerate(LE_energies_nu_minus_one_l):
                    for p_1, LE_energy_1 in enumerate(LE_energies_1_l):
                        if b_nu_minus_one_l_tuple[p_nu_minus_one, -1] > b_1_l[p_1, 0]: continue
                        if LE_energy_nu_minus_one + LE_energy_1 > E_max[nu]: continue
                        LE_energies[nu][l_tuple_nu].append(LE_energy_nu_minus_one + LE_energy_1)
                        combine_indices_l_tuple_nu_minus_one.append(p_nu_minus_one) 
                        combine_indices_l_tuple_1.append(p_1)
                        b_nu[l_tuple_nu].append(torch.concatenate([b_nu_minus_one_l_tuple[p_nu_minus_one, :], b_1_l[p_1, :]]))

                if len(b_nu[l_tuple_nu]) == 0: 
                    LE_energies[nu].pop(l_tuple_nu)
                    b_nu.pop(l_tuple_nu)
                    continue
                b_nu[l_tuple_nu] = torch.stack(b_nu[l_tuple_nu])
                LE_energies[nu][l_tuple_nu] = torch.stack(LE_energies[nu][l_tuple_nu])
                combine_indices_l_tuple_nu_minus_one = torch.tensor(combine_indices_l_tuple_nu_minus_one, dtype=torch.long, device=device)
                combine_indices_l_tuple_1 = torch.tensor(combine_indices_l_tuple_1, dtype=torch.long, device=device)
                combine_indices[nu][l_tuple_nu] = (combine_indices_l_tuple_nu_minus_one, combine_indices_l_tuple_1)

                multiplicities[nu][l_tuple_nu] = []
                for b_nu_feat in b_nu[l_tuple_nu]:
                    _, counts = torch.unique(b_nu_feat, return_counts=True)
                    multiplicity = math.factorial(nu)
                    for count in counts:
                        multiplicity = multiplicity/math.factorial(count.item())
                    multiplicity = np.sqrt(multiplicity)
                    multiplicities[nu][l_tuple_nu].append(multiplicity)
                multiplicities[nu][l_tuple_nu] = torch.tensor(multiplicities[nu][l_tuple_nu], dtype=torch.get_default_dtype(), device=device)

        b_nu_minus_one = b_nu

    return combine_indices, multiplicities, LE_energies
