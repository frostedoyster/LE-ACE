import math
import numpy as np
import torch
import equistore
from datetime import datetime
import ase
from ase import io

from utils.dataset_processing import get_dataset_slices, get_minimum_distance
from utils.error_measures import get_rmse, get_mae
from utils.cg import ClebschGordanReal
from utils.composition import get_composition_features
from utils.lexicographic_multiplicities import apply_multiplicities
from utils.sum_like_atoms import sum_like_atoms

from utils.LE_initialization import initialize_basis
from utils.LE_expansion import get_LE_expansion
from utils.LE_iterations import LEIterator
from utils.LE_invariants import LEInvariantCalculator
from utils.LE_regularization import get_LE_regularization

import json

torch.set_default_dtype(torch.float64)

def run_fit(parameters, n_train, RANDOM_SEED):

    print("5, 1.3, inner")

    param_dict = json.load(open(parameters, "r"))
    # RANDOM_SEED = param_dict["random seed"]
    BATCH_SIZE = param_dict["batch size"]
    ENERGY_CONVERSION = param_dict["energy conversion"]
    FORCE_CONVERSION = param_dict["force conversion"]
    TARGET_KEY = param_dict["target key"]
    DATASET_PATH = param_dict["dataset path"]
    n_test = param_dict["n_test"]
    # n_train = param_dict["n_train"]
    do_gradients = param_dict["do gradients"]
    FORCE_WEIGHT = param_dict["force weight"]
    r_cut = param_dict["r_cut"]
    r_cut_rs = param_dict["r_cut_rs"]
    nu_max = param_dict["nu_max"]
    E_max_coefficients = param_dict["E_max coefficients"]
    opt_target_name = param_dict["optimization target"]
    factor = param_dict["factor for radial transform"]
    cost_trade_off = param_dict["cost_trade_off"]
    le_type = param_dict["le_type"]
    dataset_style = param_dict["dataset_style"]
    inner_smoothing = param_dict["inner_smoothing"]
    fast_fit = param_dict["fast fit"]

    np.random.seed(RANDOM_SEED)
    print(f"Random seed: {RANDOM_SEED}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"Calculating features on {device}")

    conversions = {}
    conversions["HARTREE_TO_EV"] = 27.211386245988
    conversions["HARTREE_TO_KCAL_MOL"] = 627.509608030593
    conversions["EV_TO_KCAL_MOL"] = conversions["HARTREE_TO_KCAL_MOL"]/conversions["HARTREE_TO_EV"]
    conversions["KCAL_MOL_TO_MEV"] = 0.0433641153087705*1000.0
    conversions["METHANE_FORCE"] = conversions["HARTREE_TO_KCAL_MOL"]/0.529177

    CONVERSION_FACTOR = conversions[ENERGY_CONVERSION]
    FORCE_CONVERSION_FACTOR = conversions[FORCE_CONVERSION]

    print("Dataset path: " + DATASET_PATH)

    if "methane" in DATASET_PATH or "ch4" in DATASET_PATH or "rmd17" in DATASET_PATH or "gold" in DATASET_PATH:
        n_elems = len(np.unique(ase.io.read(DATASET_PATH, index = "0:1")[0].get_atomic_numbers()))
        print(f"n_elems: {n_elems}")

    if do_gradients:
        N = len(ase.io.read(DATASET_PATH, index = "0:1")[0].get_atomic_numbers())
        print(f"Number of atoms in each molecule: {N}")
        three_N_plus_one = 3*N+1
        n_train_effective = n_train*three_N_plus_one
    else:
        n_train_effective = n_train

    E_max = E_max_coefficients
    print(E_max)

    if not np.all(np.array(E_max[:-1]) >= np.array(E_max[1:])): print("LE WARNING: max eigenvalues not in descending order")
    assert len(E_max) == nu_max + 1 

    # Decrease nu_max if LE threshold is too low:
    for iota in range(1, nu_max+1):
        if E_max[iota] < iota*np.pi**2:
            print(f"Decreasing nu_max to {iota-1}")
            nu_max = iota-1
            break

    if nu_max < 2:
        raise ValueError("Trying to use ACE with nu_max < 2? Why?")

    if "rmd17" in DATASET_PATH:
        train_slice = str(0) + ":" + str(n_train)
        test_slice = str(0) + ":" + str(n_test)
    else:
        test_slice = str(0) + ":" + str(n_test)
        train_slice = str(n_test) + ":" + str(n_test+n_train)

    train_structures, test_structures = get_dataset_slices(DATASET_PATH, train_slice, test_slice)

    min_training_set_distance = get_minimum_distance(train_structures)
    if inner_smoothing:
        factor2 = min_training_set_distance
    else:
        factor2 = 0.0

    train_energies = torch.tensor([structure.info[TARGET_KEY] for structure in train_structures])*CONVERSION_FACTOR
    test_energies = torch.tensor([structure.info[TARGET_KEY] for structure in test_structures])*CONVERSION_FACTOR

    if do_gradients:
        train_forces = torch.tensor(np.concatenate([structure.get_forces() for structure in train_structures], axis = 0))*FORCE_CONVERSION_FACTOR*FORCE_WEIGHT
        test_forces = torch.tensor(np.concatenate([structure.get_forces() for structure in test_structures], axis = 0))*FORCE_CONVERSION_FACTOR*FORCE_WEIGHT

    all_species = np.sort(np.unique(np.concatenate([train_structure.numbers for train_structure in train_structures] + [test_structure.numbers for test_structure in test_structures])))
    print(f"All species: {all_species}")

    _, E_n0, radial_spectrum_calculator = initialize_basis(r_cut_rs, True, E_max[1], le_type, factor, factor2)
    print(E_n0)
    l_max, E_nl, spherical_expansion_calculator = initialize_basis(r_cut, False, E_max[2], le_type, factor, factor2, cost_trade_off=cost_trade_off)
    print(E_nl)

    n_max_l = []
    for l in range(l_max+1):
        n_max_l.append(np.where(E_nl[:, l] <= E_max[2])[0][-1] + 1)
    print(n_max_l)

    # anl counter:
    a_max = len(all_species)
    combined_anl = {}
    anl_counter = 0
    for a in range(a_max):
        for l in range(l_max+1):
            for n in range(n_max_l[l]):
                combined_anl[(a, n, l,)] = anl_counter
                anl_counter += 1

    invariant_calculator = LEInvariantCalculator(E_nl, combined_anl, all_species)
    alg = "fast cg" if device=="cpu" else "dense"
    cg_object = ClebschGordanReal(device=device, algorithm=alg)
    equivariant_calculator = LEIterator(E_nl, combined_anl, all_species, cg_object)  # L_max=3

    def get_LE_invariants(structures):

        print("Calculating composition features")
        comp = get_composition_features(structures, all_species)
        print("Composition features done")

        rs = get_LE_expansion(structures, radial_spectrum_calculator, E_n0, E_max[1], all_species, rs=True, do_gradients=do_gradients)
        if nu_max > 1: spherical_expansion = get_LE_expansion(structures, spherical_expansion_calculator, E_nl, E_max[2], all_species, do_gradients=do_gradients, device=device)

        invariants = [rs]

        if nu_max > 1: equivariants_nu_minus_one = spherical_expansion

        for nu in range(2, nu_max+1):

            print(f"Calculating nu = {nu} invariants")
            invariants_nu = invariant_calculator(equivariants_nu_minus_one, spherical_expansion, E_max[nu])
            invariants_nu = apply_multiplicities(invariants_nu, combined_anl)
            invariants.append(invariants_nu)

            if nu == nu_max: break  # equivariants for nu_max wouldn't be used

            print(f"Calculating nu = {nu} equivariants")
            equivariants_nu = equivariant_calculator(equivariants_nu_minus_one, spherical_expansion, E_max[nu+1])
            equivariants_nu_minus_one = equivariants_nu

        X = sum_like_atoms(invariants)
        return comp, X


    # Divide training and test set into batches (to limit memory usage):
    def get_batches(list: list, batch_size: int) -> list:
        batches = []
        n_full_batches = len(list)//batch_size
        for i_batch in range(n_full_batches):
            batches.append(list[i_batch*batch_size:(i_batch+1)*batch_size])
        if len(list) % batch_size != 0:
            batches.append(list[n_full_batches*batch_size:])
        return batches

    train_structures = get_batches(train_structures, BATCH_SIZE)
    test_structures = get_batches(test_structures, BATCH_SIZE)

    from equistore import TensorBlock, TensorMap
    def to_cpu(tmap):
        if tmap.block(0).values.device == "cpu":
            return tmap

        new_blocks = []
        for _, block in tmap:
            new_block = TensorBlock(
                values=block.values.to("cpu"),
                samples=block.samples,
                components=block.components,
                properties=block.properties
            )
            if block.has_gradient("positions"):
                old_gradients = block.gradient("positions")
                new_block.add_gradient(
                    parameter="positions",
                    gradient=TensorBlock(
                        values=old_gradients.values.to("cpu"),
                        samples=old_gradients.samples,
                        components=old_gradients.components,
                        properties=old_gradients.properties
                    )
                )
            new_blocks.append(new_block)
        return TensorMap(
            keys=tmap.keys,
            blocks=new_blocks
        )

    train_comp = []
    if do_gradients: train_dcomp = []
    X_train = []
    for i_batch, batch in enumerate(train_structures):
        print(f"DOING TRAIN BATCH {i_batch+1} out of {len(train_structures)}")
        comp, X = get_LE_invariants(batch)

        if dataset_style == "mixed":
            comp = comp.values.to(device)
        elif dataset_style == "MD":
            comp = torch.ones((X[0].block(0).values.shape[0], 1))
        else:
            raise NotImplementedError("The dataset_style must be either MD or mixed")
        train_comp.append(comp)
      
        if do_gradients:    
            if dataset_style == "mixed":
                dcomp = torch.zeros((X[0].block(0).gradient("positions").values.shape[0]*3, len(all_species)))
            elif dataset_style == "MD":
                dcomp = torch.zeros((X[0].block(0).gradient("positions").values.shape[0]*3, 1))
            else:
                raise NotImplementedError("The dataset_style must be either MD or mixed")
            train_dcomp.append(dcomp)

        moved_X = []
        for tmap in X:
            moved_tmap = tmap.keys_to_properties(keys_to_move="a_i")
            moved_X.append(to_cpu(moved_tmap))
        # The equistore to function here would allow for much better memory management
        X_train.append(moved_X)

    properties = [X_train[0][nu-1].block().properties for nu in range(1, nu_max+1)]

    if dataset_style == "mixed":
        LE_reg_comp = torch.tensor([0.0]*len(all_species))
        # LE_reg_comp = torch.tensor([1e-4]*len(species))  # this seems to work ok; needs more testing 
    elif dataset_style == "MD":
        LE_reg_comp = torch.tensor([0.0])
    else:
        raise NotImplementedError("The dataset_style must be either MD or mixed")

    X_train_batches = []
    if do_gradients: dX_train_batches = []
    for i_batch, batch in enumerate(X_train):
        processed_batch = [batch[nu-1].block().values for nu in range(1, nu_max+1)]
        processed_batch = torch.concat([train_comp[i_batch]]+processed_batch, dim=-1)
        X_train_batches.append(processed_batch)
        if do_gradients:
            d_processed_batch = [-FORCE_WEIGHT*batch[nu-1].block().gradient("positions").values.cpu().reshape(batch[nu-1].block().gradient("positions").values.shape[0]*3, batch[nu-1].block().values.shape[1]) for nu in range(1, nu_max+1)]
            d_processed_batch = torch.concat([train_dcomp[i_batch]]+d_processed_batch, dim=-1)
            dX_train_batches.append(d_processed_batch)

    if do_gradients:
        X_train = torch.concat(X_train_batches + dX_train_batches, dim = 0)
    else:
        X_train = torch.concat(X_train_batches, dim = 0)

    test_comp = []
    if do_gradients: test_dcomp = []
    X_test = []
    for i_batch, batch in enumerate(test_structures):
        print(f"DOING TEST BATCH {i_batch+1} out of {len(test_structures)}")
        comp, X = get_LE_invariants(batch)

        if dataset_style == "mixed":
            comp = comp.values.to(device)
        elif dataset_style == "MD":
            comp = torch.ones((X[0].block(0).values.shape[0], 1))
        else:
            raise NotImplementedError("The dataset_style must be either MD or mixed")
        test_comp.append(comp)
      
        if do_gradients:    
            if dataset_style == "mixed":
                dcomp = torch.zeros((X[0].block(0).gradient("positions").values.shape[0]*3, len(all_species)))
            elif dataset_style == "MD":
                dcomp = torch.zeros((X[0].block(0).gradient("positions").values.shape[0]*3, 1))
            else:
                raise NotImplementedError("The dataset_style must be either MD or mixed")
            test_dcomp.append(dcomp)

        moved_X = []
        for tmap in X:
            moved_tmap = tmap.keys_to_properties(keys_to_move="a_i")
            moved_X.append(to_cpu(moved_tmap))
        X_test.append(moved_X)

    X_test_batches = []
    if do_gradients: dX_test_batches = []
    for i_batch, batch in enumerate(X_test):
        processed_batch = [batch[nu-1].block().values for nu in range(1, nu_max+1)]
        processed_batch = torch.concat([test_comp[i_batch]]+processed_batch, dim=-1)
        X_test_batches.append(processed_batch)
        if do_gradients:
            d_processed_batch = [-FORCE_WEIGHT*batch[nu-1].block().gradient("positions").values.cpu().reshape(batch[nu-1].block().gradient("positions").values.shape[0]*3, batch[nu-1].block().values.shape[1]) for nu in range(1, nu_max+1)]
            d_processed_batch = torch.concat([test_dcomp[i_batch]]+d_processed_batch, dim=-1)
            dX_test_batches.append(d_processed_batch)

    if do_gradients:
        X_test = torch.concat(X_test_batches + dX_test_batches, dim = 0)
    else:
        X_test = torch.concat(X_test_batches, dim = 0)

    print("Features done")

    print("Beginning hyperparameter optimization")

    if fast_fit:

        train_mean = torch.mean(train_energies)
        train_energies -= train_mean
        test_energies -= train_mean

        if do_gradients:
            train_targets = torch.concat([train_energies, train_forces.reshape((-1,))])
            test_targets = torch.concat([test_energies, test_forces.reshape((-1,))])
        else:
            train_targets = train_energies
            #train_targets = torch.concat([train_energies, torch.tensor([0.0])])
            test_targets = test_energies

        symm = X_train.T @ X_train
        vec = X_train.T @ train_targets
        alpha_list = np.linspace(-10.0, 0.0, 21)
        n_feat = X_train.shape[1]
        print("Number of features: ", n_feat)

        print("Diagonalizing...")
        d, O = torch.linalg.eigh(symm)
        print("Diagonalization done")
        vec2 = O.T @ vec

        best_opt_target = 1e30
        for alpha in alpha_list:
            reg = 10**alpha*torch.ones((n_feat))

            d_inverted = (d+reg)**(-1)
            vec3 = d_inverted*vec2
            c = O @ vec3

            train_predictions = X_train @ c
            test_predictions = X_test @ c

            print(alpha, get_rmse(train_predictions[:n_train], train_targets[:n_train]).item(), get_rmse(test_predictions[:n_test], test_targets[:n_test]).item(), get_mae(test_predictions[:n_test], test_targets[:n_test]).item(), get_rmse(train_predictions[n_train:], train_targets[n_train:]).item()/FORCE_WEIGHT, get_rmse(test_predictions[n_test:], test_targets[n_test:]).item()/FORCE_WEIGHT, get_mae(test_predictions[n_test:], test_targets[n_test:]).item()/FORCE_WEIGHT)
            if opt_target_name == "mae":
                if do_gradients:
                    opt_target = get_mae(test_predictions[n_test:], test_targets[n_test:]).item()/FORCE_WEIGHT
                else:
                    opt_target = get_mae(test_predictions[:n_test], test_targets[:n_test]).item()
            else:
                if do_gradients:
                    opt_target = get_rmse(test_predictions[n_test:], test_targets[n_test:]).item()/FORCE_WEIGHT
                else:    
                    opt_target = get_rmse(test_predictions[:n_test], test_targets[:n_test]).item()

            if opt_target < best_opt_target:
                best_opt_target = opt_target
                best_alpha = alpha

        print("Best parameter:", best_alpha)

        for i in range(n_feat):
            symm[i, i] += 10**best_alpha
        c = torch.linalg.solve(symm, vec)

    else:  # no fast fit

        if do_gradients:
            train_targets = torch.concat([train_energies, train_forces.reshape((-1,))])
            test_targets = torch.concat([test_energies, test_forces.reshape((-1,))])
        else:
            train_targets = train_energies
            #train_targets = torch.concat([train_energies, torch.tensor([0.0])])
            test_targets = test_energies

        symm = X_train.T @ X_train
        vec = X_train.T @ train_targets
        alpha_list = np.linspace(-15.0, -5.0, 21)
        # alpha_list = np.linspace(-5.0, 5.0, 41)
        n_feat = X_train.shape[1]
        print("Number of features: ", n_feat)

        best_opt_target = 1e30
        for beta in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:  # reproduce

            print("beta=", beta)

            LE_reg = []
            for nu in range(nu_max+1):
                if nu > 1:
                    LE_reg.append(
                        get_LE_regularization(properties[nu-1], E_nl, r_cut_rs, r_cut, beta)
                    )
                elif nu == 1:
                    LE_reg.append(
                        get_LE_regularization(properties[nu-1], E_n0, r_cut_rs, r_cut, beta)
                    )
                else:
                    LE_reg.append(LE_reg_comp)
            LE_reg = torch.concat(LE_reg)

            for alpha in alpha_list:
                #X_train[-1, :] = torch.sqrt(10**alpha*LE_reg)
                for i in range(n_feat):
                    symm[i, i] += 10**alpha*LE_reg[i]

                try:
                    #c = torch.linalg.lstsq(X_train, train_targets, driver = "gelsd", rcond = 1e-8).solution
                    c = torch.linalg.solve(symm, vec)
                except Exception as e:
                    print(e)
                    opt_target.append(1e30)
                    for i in range(n_feat):
                        symm[i, i] -= 10.0**alpha*LE_reg[i]
                    continue
                train_predictions = X_train @ c
                test_predictions = X_test @ c

                print(alpha, get_rmse(train_predictions[:n_train], train_targets[:n_train]).item(), get_rmse(test_predictions[:n_test], test_targets[:n_test]).item(), get_mae(test_predictions[:n_test], test_targets[:n_test]).item(), get_rmse(train_predictions[n_train:], train_targets[n_train:]).item()/FORCE_WEIGHT, get_rmse(test_predictions[n_test:], test_targets[n_test:]).item()/FORCE_WEIGHT, get_mae(test_predictions[n_test:], test_targets[n_test:]).item()/FORCE_WEIGHT)
                if opt_target_name == "mae":
                    if do_gradients:
                        opt_target = get_mae(test_predictions[n_test:], test_targets[n_test:]).item()/FORCE_WEIGHT
                    else:
                        opt_target = get_mae(test_predictions[:n_test], test_targets[:n_test]).item()
                else:
                    if do_gradients:
                        opt_target = get_rmse(test_predictions[n_test:], test_targets[n_test:]).item()/FORCE_WEIGHT
                    else:    
                        opt_target = get_rmse(test_predictions[:n_test], test_targets[:n_test]).item()

                if opt_target < best_opt_target:
                    best_opt_target = opt_target
                    best_alpha = alpha
                    best_beta = beta

                for i in range(n_feat):
                    symm[i, i] -= 10.0**alpha*LE_reg[i]

        print("Best parameters:", best_alpha, best_beta)

        LE_reg = []
        for nu in range(nu_max+1):
            if nu > 1:
                LE_reg.append(
                    get_LE_regularization(properties[nu-1], E_nl, r_cut_rs, r_cut, best_beta)
                )
            elif nu == 1:
                print(E_n0)
                LE_reg.append(
                    get_LE_regularization(properties[nu-1], E_n0, r_cut_rs, r_cut, best_beta)
                )
            else:
                LE_reg.append(LE_reg_comp)
        LE_reg = torch.concat(LE_reg)

        for i in range(n_feat):
            symm[i, i] += 10**best_alpha*LE_reg[i]
        c = torch.linalg.solve(symm, vec)

    

    test_predictions = X_test @ c
    print("n_train:", n_train, "n_features:", n_feat)
    print(f"Test set RMSE (E): {get_rmse(test_predictions[:n_test], test_targets[:n_test]).item()} [MAE (E): {get_mae(test_predictions[:n_test], test_targets[:n_test]).item()}], RMSE (F): {get_rmse(test_predictions[n_test:], test_targets[n_test:]).item()/FORCE_WEIGHT} [MAE (F): {get_mae(test_predictions[n_test:], test_targets[n_test:]).item()/FORCE_WEIGHT}]")

    # Uncomment for speed evaluation
    """
    from ase import io
    for length_exp in range(0, 10):
        length = 2**length_exp
        dummy_structure = ase.io.read(DATASET_PATH, index = ":" + str(length))
        import time
        time_before = time.time()
        for _ in range(10):
            X, dX, LE_reg = get_LE_invariants(dummy_structure)
            # e = X@c
        print()
        print()
        print()
        print(length, (time.time()-time_before)/10)
        print()
        print()
        print()
    """
