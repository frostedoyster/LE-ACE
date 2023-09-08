import argparse
import json
import math
import numpy as np
import torch
import metatensor.torch
from datetime import datetime
import ase
from ase import io

from LE_ACE.dataset_processing import get_dataset_slices, get_minimum_distance
from LE_ACE import LE_ACE

torch.set_default_dtype(torch.float64)


def run_fit(parameters, n_train, RANDOM_SEED):

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
    if dataset_style == "MD":
        fixed_stoichiometry = True
    else:
        fixed_stoichiometry = False
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

    ENERGY_CONVERSION_FACTOR = conversions[ENERGY_CONVERSION]
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

    all_species = np.sort(np.unique(np.concatenate([train_structure.numbers for train_structure in train_structures] + [test_structure.numbers for test_structure in test_structures])))

    min_training_set_distance = get_minimum_distance(train_structures)
    if inner_smoothing:
        factor2 = min_training_set_distance
    else:
        factor2 = 0.0

    le_ace = LE_ACE(
        r_cut_rs=r_cut_rs,
        r_cut=r_cut,
        E_max=E_max,
        all_species=all_species,
        le_type=le_type,
        factor=factor,
        factor2=factor2,
        cost_trade_off=cost_trade_off,
        fixed_stoichiometry=fixed_stoichiometry,
        is_trace=is_trace,
        n_trace=n_trace,
        device=device
    )

    accuracy_dict = le_ace.train(
        train_structures=train_structures,
        validation_structures=test_structures,
        do_gradients=do_gradients,
        opt_target_name=opt_target_name,
    )

    print(ENERGY_CONVERSION_FACTOR*accuracy_dict["validation RMSE energies"])
    print(ENERGY_CONVERSION_FACTOR*accuracy_dict["validation MAE energies"])
    print(FORCE_CONVERSION_FACTOR*accuracy_dict["validation RMSE forces"])
    print(FORCE_CONVERSION_FACTOR*accuracy_dict["validation MAE forces"])



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
        help="The file containing the parameters. JSON formatted dictionary.",
    )

    parser.add_argument(
        "random_seed",
        type=int,
        help="The file containing the parameters. JSON formatted dictionary.",
    )

    args = parser.parse_args()
    run_fit(args.parameters, args.n_train, args.random_seed)
