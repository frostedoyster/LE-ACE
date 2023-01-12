from utils.LE_expansion import get_LE_expansion, write_spline

import math
import torch
import numpy as np
from datetime import datetime
import ase
from ase import io

from utils.dataset_processing import get_dataset_slices, get_minimum_distance
from utils.error_measures import get_rmse, get_mae
from utils.cg import ClebschGordanReal
from utils.composition import get_composition_features
from utils.lexicographic_multiplicities import apply_multiplicities
from utils.spherical_bessel_zeros import Jn_zeros, get_laplacian_eigenvalues
from utils.sum_like_atoms import sum_like_atoms

from utils.LE_iterations import LEIterator
from utils.LE_invariants import LEInvariantCalculator

import os
import json

torch.set_default_dtype(torch.float64)

def run_fit(parameters):

    param_dict = json.load(open(parameters, "r"))
    RANDOM_SEED = param_dict["random seed"]
    BATCH_SIZE = param_dict["batch size"]
    ENERGY_CONVERSION = param_dict["energy conversion"]
    FORCE_CONVERSION = param_dict["force conversion"]
    TARGET_KEY = param_dict["target key"]
    DATASET_PATH = param_dict["dataset path"]
    n_test = param_dict["n_test"]
    n_train = param_dict["n_train"]
    do_gradients = param_dict["do gradients"]
    global r_cut
    r_cut = param_dict["r_cut"]
    global r_cut_rs
    r_cut_rs = param_dict["r_cut_rs"]
    nu_max = param_dict["nu_max"]
    E_max_coefficients = param_dict["E_max coefficients"]
    opt_target_name = param_dict["optimization target"]
    global factor 
    factor = param_dict["factor for radial transform"]
    L_max = param_dict["L_max"]
    print(L_max)

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
    conversions["NO_CONVERSION"] = 1.0

    CONVERSION_FACTOR = conversions[ENERGY_CONVERSION]
    FORCE_CONVERSION_FACTOR = conversions[FORCE_CONVERSION]

    print("Dataset path: " + DATASET_PATH)
    FORCE_WEIGHT = 1.0

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

    element_factors = [0.0] + [math.factorial(nu+n_elems)/(math.factorial(nu+1)*math.factorial(n_elems-1)) for nu in range(1, nu_max+1)]

    print(E_max_coefficients)
    E_max = [1e30]
    for iota in range(len((E_max_coefficients))):
        if iota == 0: continue
        E_max.append(E_max_coefficients[iota]*(n_train_effective/element_factors[iota])**(2.0/(3.0*iota)))
    print(E_max)

    if not np.all(np.array(E_max[:-1]) >= np.array(E_max[1:])): print("LE WARNING: max eigenvalues not in descending order")
    assert len(E_max) == nu_max + 1 

    # Decrease nu_max if LE threshold is too low:
    for iota in range(1, nu_max+1):
        if E_max[iota] < iota*np.pi**2:
            print(f"Decreasing nu_max to {iota-1}")
            nu_max = iota-1
            break

    if "rmd17" in DATASET_PATH:
        train_slice = str(0) + ":" + str(n_train)
        test_slice = str(0) + ":" + str(n_test)
    else:
        test_slice = str(0) + ":" + str(n_test)
        train_slice = str(n_test) + ":" + str(n_test+n_train)

    train_structures, test_structures = get_dataset_slices(DATASET_PATH, train_slice, test_slice)

    min_training_set_distance = get_minimum_distance(train_structures)
    global factor2
    factor2 = 0.9*min_training_set_distance

    train_energies = torch.tensor([structure.info[TARGET_KEY] for structure in train_structures])*CONVERSION_FACTOR
    test_energies = torch.tensor([structure.info[TARGET_KEY] for structure in test_structures])*CONVERSION_FACTOR

    if do_gradients:
        train_forces = torch.tensor(np.concatenate([structure.get_forces() for structure in train_structures], axis = 0))*FORCE_CONVERSION_FACTOR*FORCE_WEIGHT
        test_forces = torch.tensor(np.concatenate([structure.get_forces() for structure in test_structures], axis = 0))*FORCE_CONVERSION_FACTOR*FORCE_WEIGHT

    E_nl = get_laplacian_eigenvalues(50, 50)

    all_species = np.sort(np.unique(np.concatenate([train_structure.numbers for train_structure in train_structures] + [test_structure.numbers for test_structure in test_structures])))
    print(f"All species: {all_species}")

    n_max_rs = np.where(E_nl[:, 0] <= E_max[1])[0][-1] + 1
    print(f"Radial spectrum: n_max = {n_max_rs}")

    date_time = datetime.now()
    date_time = date_time.strftime("%m-%d-%Y-%H-%M-%S-%f")
    rs_spline_path = "splines/splines-rs-" + date_time + ".txt"
    write_spline(r_cut_rs, n_max_rs, 0, rs_spline_path)

    if nu_max > 1:
        n_max = np.where(E_nl[:, 0] <= E_max[2])[0][-1] + 1
        l_max = np.where(E_nl[0, :] <= E_max[2])[0][-1]
        print(f"Spherical expansion: n_max = {n_max}, l_max = {l_max}")

        # anl counter:
        a_max = len(all_species)
        n_max_l = []
        a_i = all_species[0]
        for l in range(l_max+1):
            n_max_l.append(np.where(E_nl[:, l] <= E_max[2])[0][-1] + 1)
        combined_anl = {}
        anl_counter = 0
        for a in range(a_max):
            for l in range(l_max+1):
                for n in range(n_max_l[l]):
                    combined_anl[(a, n, l,)] = anl_counter
                    anl_counter += 1

    spline_path = "splines/splines-" + date_time + ".txt"
    write_spline(r_cut, n_max, l_max, spline_path)

    invariant_calculator = LEInvariantCalculator(E_nl, combined_anl, all_species)
    cg_object = ClebschGordanReal()
    equivariant_calculator = LEIterator(E_nl, combined_anl, all_species, cg_object, L_max=L_max)

    def get_LE_invariants(structures):

        print("Calculating composition features")
        comp = get_composition_features(structures, all_species)
        print("Composition features done")

        rs = get_LE_expansion(structures, rs_spline_path, E_nl, E_max[1], r_cut_rs, all_species, rs=True, do_gradients=do_gradients)
        if nu_max > 1: spherical_expansion = get_LE_expansion(structures, spline_path, E_nl, E_max[2], r_cut, all_species, do_gradients=do_gradients, device=device)

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

        X, dX, LE_reg = sum_like_atoms(comp, invariants, all_species, E_nl)
        return X, dX, LE_reg


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

    X_train = []
    if do_gradients: dX_train = []
    for i_batch, batch in enumerate(train_structures):
        print(f"DOING TRAIN BATCH {i_batch+1} out of {len(train_structures)}")
        X, dX, LE_reg = get_LE_invariants(batch)
        X_train.append(X)
        if do_gradients: dX_train.append(-FORCE_WEIGHT*dX.reshape(dX.shape[0]*3, dX.shape[2]))  # note the minus sign

    if do_gradients:
        X_train = torch.concat(dX_train, dim = 0)
    else:
        X_train = torch.concat(X_train, dim = 0)
        # X_train = torch.concat(X_train + [torch.zeros((1, X_train[0].shape[1]))], dim = 0)
    #X_train = torch.concat([X_train, torch.zeros((1, X_train.shape[1]))], dim = 0)
    #print(X_train.shape)

    X_test = []
    if do_gradients: dX_test = []
    for i_batch, batch in enumerate(test_structures):
        print(f"DOING TEST BATCH {i_batch+1} out of {len(test_structures)}")
        X, dX, LE_reg = get_LE_invariants(batch)
        X_test.append(X)
        if do_gradients: dX_test.append(-FORCE_WEIGHT*dX.reshape(dX.shape[0]*3, dX.shape[2]))  # note the minus sign

    if do_gradients:
        X_test = torch.concat(dX_test, dim = 0)
    else:
        X_test = torch.concat(X_test, dim = 0)

    print("Features done")

    print("Beginning hyperparameter optimization")

    if do_gradients:
        train_targets = train_forces.reshape((-1,))
        test_targets = test_forces.reshape((-1,))
    else:
        train_targets = train_energies
        #train_targets = torch.concat([train_energies, torch.tensor([0.0])])
        test_targets = test_energies

    symm = X_train.T @ X_train
    vec = X_train.T @ train_targets
    opt_target = []
    alpha_list = np.linspace(-16.0, -6.0, 41)
    # alpha_list = np.linspace(-5.0, 5.0, 41)
    n_feat = X_train.shape[1]
    print("Number of features: ", n_feat)

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
            continue
        train_predictions = X_train @ c
        test_predictions = X_test @ c
        print(alpha, get_rmse(train_predictions, train_targets).item()/FORCE_WEIGHT, get_mae(train_predictions, train_targets).item()/FORCE_WEIGHT, get_rmse(test_predictions, test_targets).item()/FORCE_WEIGHT, get_mae(test_predictions, test_targets).item()/FORCE_WEIGHT)
        if opt_target_name == "mae":
            opt_target.append(get_mae(test_predictions, test_targets).item())
        else:
            opt_target.append(get_rmse(test_predictions, test_targets).item())

        for i in range(n_feat):
            symm[i, i] -= 10.0**alpha*LE_reg[i]

    best_alpha = alpha_list[np.argmin(opt_target)]
    print(best_alpha, np.min(opt_target))

    for i in range(n_feat):
        symm[i, i] += 10**best_alpha*LE_reg[i]
    c = torch.linalg.solve(symm, vec)

    #X_train[-1, :] = torch.sqrt(10**best_alpha*LE_reg)
    #c = torch.linalg.lstsq(X_train, train_targets, driver = "gelsd", rcond = 1e-8).solution

    test_predictions = X_test @ c
    print("n_train:", n_train, "n_features:", n_feat)
    print(f"Test error RMSE (F): {get_rmse(test_predictions, test_targets).item()/FORCE_WEIGHT} [MAE (F): {get_mae(test_predictions, test_targets).item()/FORCE_WEIGHT}]")
    print(f"Percentage test RMSE: {get_rmse(test_predictions, test_targets).item()/get_rmse(torch.zeros_like(test_targets), test_targets).item()*100.0}%")
    # print(c[0])

    # Speed evaluation
    # """
    from ase import io
    length = 1
    dummy_structure = ase.io.read(DATASET_PATH, index = ":" + str(length))
    import time
    time_before = time.time()
    for _ in range(10):
        X, dX, LE_reg = get_LE_invariants(dummy_structure)
        forces = - dX @ c
    print()
    print()
    print(f"Percentage test RMSE: {get_rmse(test_predictions, test_targets).item()/get_rmse(torch.zeros_like(test_targets), test_targets).item()*100.0}%")
    print("Estimated time per MD step: ", (time.time()-time_before)/10)
    print()
    print()
    # """

    os.remove(rs_spline_path)
    os.remove(spline_path)
