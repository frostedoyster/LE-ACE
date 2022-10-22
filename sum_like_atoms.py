import numpy as np
import torch
from LE_regularization import get_LE_regularization

def sum_like_atoms(comp, invariants, species, E_nl):

    n_structures = len(comp.samples["structure"])
    n_features = [invariants_nu.block(0).values.shape[1] for invariants_nu in invariants]  # Assuming same number of features for each center block at a given correlation order.

    for nu_minus_one in range(len(invariants)):
        print(f"nu = {nu_minus_one+1}:", n_features[nu_minus_one]*len(species))

    features = []
    LE_reg = []
    d_features = []
  
    for nu_minus_one in range(len(invariants)):
        for center_species in species:

            # if center_species == 1: continue  # UNCOMMENT FOR METHANE DATASET C-ONLY VERSION
            print(f"     Calculating structure features for center species {center_species}", flush = True)
            try:
                invariant_block = invariants[nu_minus_one].block(a_i=center_species)
            except ValueError:
                print("This set does not contain the above center species")
                exit()

            features_current_center_species = torch.zeros((n_structures, n_features[nu_minus_one]))

            structures = invariant_block.samples["structure"]
            len_samples = structures.shape[0]
            center_features = invariant_block.values.cpu()
            for i in range(len_samples):
                features_current_center_species[structures[i], :] += center_features[i, :]
            features.append(features_current_center_species)

            LE_reg.append(get_LE_regularization(invariant_block.properties, E_nl))

            if invariant_block.has_gradient("positions"):
                gradients = invariant_block.gradient("positions")
                force_centers = gradients.samples[["structure", "atom"]]
                unique_force_centers = np.unique(force_centers)
                unique_force_centers = [(unique_force_center[0], unique_force_center[1]) for unique_force_center in unique_force_centers]  # make list of tuples which are hashable
                force_centers_dict = {force_center: j for j, force_center in enumerate(unique_force_centers)}
                number_of_unique_force_centers = len(unique_force_centers)
                center_d_features = gradients.data.cpu()
                d_features_current_center_species = torch.zeros((number_of_unique_force_centers, 3, n_features[nu_minus_one]))

                force_centers = [(force_center[0], force_center[1]) for force_center in force_centers]  # transform original force centers into tuples for consistency with dict

                len_grad_samples = gradients.data.shape[0]
                for i in range(len_grad_samples):
                    d_features_current_center_species[force_centers_dict[force_centers[i]], :, :] += center_d_features[i, :, :]
                
                # print(d_features_current_center_species)
                d_features.append(d_features_current_center_species)

    #comp = comp.values
    #LE_reg_comp = torch.tensor([0.0]*len(all_species))

    comp = torch.ones(features[0].shape[0], 1)  # MD-like
    LE_reg_comp = torch.tensor([0.0])
    if len(d_features) != 0: d_comp = torch.zeros((d_features[0].shape[0], 3, 1))

    X = torch.concat([comp] + features, dim = -1)
    LE_reg = torch.concat([LE_reg_comp] + LE_reg, dim = -1)

    if len(d_features) != 0: 
        dX = torch.concat([d_comp] + d_features, dim = -1)
    else:
        dX = d_features

    return X, dX, LE_reg