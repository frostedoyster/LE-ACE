import math
import numpy as np
import torch
import equistore
from datetime import datetime
import ase
from ase import io
import argparse

from utils.dataset_processing import get_dataset_slices, get_minimum_distance
from utils.error_measures import get_rmse, get_mae

import json

torch.set_default_dtype(torch.float64)

def run_fit(parameters, n_train, RANDOM_SEED, datetime):

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
    is_trace = param_dict["is_trace"]
    n_trace = param_dict["n_trace"]

    np.random.seed(RANDOM_SEED)
    print(f"Random seed: {RANDOM_SEED}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"Calculating features on {device}")

    conversions = {}
    conversions["NO_CONVERSION"] = 1.0
    conversions["HARTREE_TO_EV"] = 27.211386245988
    conversions["HARTREE_TO_KCAL_MOL"] = 627.509608030593
    conversions["EV_TO_KCAL_MOL"] = conversions["HARTREE_TO_KCAL_MOL"]/conversions["HARTREE_TO_EV"]
    conversions["KCAL_MOL_TO_MEV"] = 0.0433641153087705*1000.0
    conversions["METHANE_FORCE"] = conversions["HARTREE_TO_KCAL_MOL"]/0.529177

    CONVERSION_FACTOR = conversions[ENERGY_CONVERSION]
    FORCE_CONVERSION_FACTOR = conversions[FORCE_CONVERSION]

    print("Dataset path: " + DATASET_PATH)
    print("Factor:", factor)

    if "methane" in DATASET_PATH or "ch4" in DATASET_PATH or "rmd17" in DATASET_PATH or "gold" in DATASET_PATH:
        n_elems = len(np.unique(ase.io.read(DATASET_PATH, index = "0:1")[0].get_atomic_numbers()))
        print(f"n_elems: {n_elems}")

    E_max = E_max_coefficients

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

    train_energies = torch.tensor([structure.info[TARGET_KEY] for structure in train_structures], dtype=torch.get_default_dtype(), device=device)*CONVERSION_FACTOR
    test_energies = torch.tensor([structure.info[TARGET_KEY] for structure in test_structures], dtype=torch.get_default_dtype(), device=device)*CONVERSION_FACTOR

    if do_gradients:
        train_forces = torch.tensor(np.concatenate([structure.get_forces() for structure in train_structures], axis = 0), dtype=torch.get_default_dtype(), device=device)*FORCE_CONVERSION_FACTOR*FORCE_WEIGHT
        test_forces = torch.tensor(np.concatenate([structure.get_forces() for structure in test_structures], axis = 0), dtype=torch.get_default_dtype(), device=device)*FORCE_CONVERSION_FACTOR*FORCE_WEIGHT

    all_species = np.sort(np.unique(np.concatenate([train_structure.numbers for train_structure in train_structures] + [test_structure.numbers for test_structure in test_structures])))
    n_elems = len(all_species)
    print(f"All species: {all_species}")

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

    from utils.LE_ACE import LE_ACE
    le_ace = LE_ACE(
        r_cut_rs=r_cut_rs,
        r_cut=r_cut,
        E_max=E_max,
        all_species=all_species,
        le_type=le_type,
        factor=factor,
        factor2=factor2,
        cost_trade_off=cost_trade_off,
        is_trace=is_trace,
        device=device
    )

    X_train_batches = []
    X_train_batches_grad = []
    for i_batch, batch in enumerate(train_structures):
        print(f"DOING TRAIN BATCH {i_batch+1} out of {len(train_structures)}")
        if do_gradients:
            values, gradients = le_ace.compute_with_gradients(batch)
            gradients = -FORCE_WEIGHT*gradients.reshape(gradients.shape[0]*3, values.shape[1])
            X_train_batches.append(values.cpu())
            X_train_batches_grad.append(gradients.cpu())
        else:
            values = le_ace(batch)
            X_train_batches.append(values.cpu())

    if do_gradients:
        X_train = torch.concat(X_train_batches + X_train_batches_grad, dim = 0)
    else:
        X_train = torch.concat(X_train_batches, dim = 0)

    n_feat = X_train.shape[1]
    print("Number of features: ", n_feat)
    prefix = "/scratch/izar/bigi/"
    suffix = "-" + DATASET_PATH.split("/")[-1].split(".")[0] + "-" + datetime + ".pt"
    torch.save(X_train, prefix + "X_train" + suffix)

    X_test_batches = []
    X_test_batches_grad = []
    for i_batch, batch in enumerate(test_structures):
        print(f"DOING TEST BATCH {i_batch+1} out of {len(test_structures)}")
        if do_gradients:
            values, gradients = le_ace.compute_with_gradients(batch)
            gradients = -FORCE_WEIGHT*gradients.reshape(gradients.shape[0]*3, values.shape[1])
            X_test_batches.append(values.cpu())
            X_test_batches_grad.append(gradients.cpu())
        else:
            values = le_ace(batch)
            X_test_batches.append(values.cpu())

    if do_gradients:
        X_test = torch.concat(X_test_batches + X_test_batches_grad, dim = 0)
    else:
        X_test = torch.concat(X_test_batches, dim = 0)

    torch.save(X_test, prefix + "X_test" + suffix)
    print("Features done")

    if do_gradients:
        train_targets = torch.concat([train_energies, train_forces.reshape((-1,))])
        test_targets = torch.concat([test_energies, test_forces.reshape((-1,))])
    else:
        train_targets = train_energies
        test_targets = test_energies

    torch.save(train_targets.cpu(), prefix + "y_train" + suffix)
    torch.save(test_targets.cpu(), prefix + "y_test" + suffix)
    torch.save([tensor.cpu() for tensor in le_ace.extended_LE_energies], prefix + "LE_reg" + suffix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="?"
    )

    parser.add_argument(
        "parameters",
        type=str,
        help="The file containing the parameters. JSON formatted dictionary.",
    )

    parser.add_argument(
        "n_train",
        type=int,
    )

    parser.add_argument(
        "random_seed",
        type=int,
    )

    parser.add_argument(
        "datetime",
        type=str,
    )

    args = parser.parse_args()
    run_fit(args.parameters, args.n_train, args.random_seed, args.datetime)
