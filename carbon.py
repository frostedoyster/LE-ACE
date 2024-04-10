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


def run_fit(parameters):

    param_dict = json.load(open(parameters, "r"))
    # RANDOM_SEED = param_dict["random seed"]
    BATCH_SIZE = param_dict["batch size"]
    ENERGY_CONVERSION = param_dict["energy conversion"]
    FORCE_CONVERSION = param_dict["force conversion"]
    TARGET_KEY = param_dict["target key"]
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    print("Factor:", factor)

    n_elems = 1

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

    import ase.io
    train_structures = ase.io.read("datasets/volker_carbon/train_small.xyz", ":")
    test_structures = ase.io.read("datasets/volker_carbon/test.xyz", ":")

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
        target_key=TARGET_KEY,
        force_weight=FORCE_WEIGHT,
        batch_size=BATCH_SIZE
    )

    print("validation RMSE energies", ENERGY_CONVERSION_FACTOR*accuracy_dict["validation RMSE energies"])
    print("validation MAE energies", ENERGY_CONVERSION_FACTOR*accuracy_dict["validation MAE energies"])
    if do_gradients:
        print("validation RMSE forces", FORCE_CONVERSION_FACTOR*accuracy_dict["validation RMSE forces"])
        print("validation MAE forces", FORCE_CONVERSION_FACTOR*accuracy_dict["validation MAE forces"])

    exit()

    le_ace_predictor = le_ace.get_fast_evaluator()

    import time
    from LE_ACE.structures import transform_structures
    from copy import deepcopy
    n_test = len(test_structures)
    test_structures = [test_structures[0], deepcopy(test_structures[0])]
    test_structures[1].rotate("x", 45.458)
    test_structures = transform_structures(test_structures)
    for test_structure in test_structures:
        energy = le_ace.predict([test_structure])
        print(energy)
        energy = le_ace_predictor([test_structure])
        print(energy)

    import time
    n_test = len(test_structures)
    start_time = time.time()
    for test_structure in test_structures:
        le_ace.predict([test_structure], do_positions_grad=True)
    finish_time = time.time()
    print(f"Evaluation took {(finish_time-start_time)/n_test} per structure")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="?"
    )

    parser.add_argument(
        "parameters",
        type=str,
        help="The file containing the parameters. JSON formatted dictionary.",
    )

    args = parser.parse_args()
    run_fit(args.parameters)
