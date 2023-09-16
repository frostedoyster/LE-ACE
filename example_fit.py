import numpy as np
import torch
import ase
import ase.io

from LE_ACE import LE_ACE

torch.set_default_dtype(torch.float64)

# Model hyperparameters
r_cut = 6.0
r_cut_rs = 6.0
le_type = "physical"
factor = 2.0
E_max = [-1, 3000.0, 600.0, 300.0]
device = "cuda" if torch.cuda.is_available() else "cpu"

# Training options:
do_gradients = True
opt_target_name = "rmse"
target_key = "free_energy"
force_weight = 0.1
batch_size = 100

train_structures = (
    ase.io.read("datasets/Si_isolated_atom_DFT.xyz", ":") +
    ase.io.read("datasets/Si_dimers_DFT.xyz", ":") +
    ase.io.read("datasets/Si_trimers_DFT.xyz", ":") +
    ase.io.read("datasets/Si_tetramers_DFT.xyz", ":")
)
test_structures = ase.io.read("datasets/Si_pentamers_DFT.xyz", ":")

all_species = np.sort(np.unique(np.concatenate([train_structure.numbers for train_structure in train_structures] + [test_structure.numbers for test_structure in test_structures])))

le_ace = LE_ACE(
    r_cut_rs=r_cut_rs,
    r_cut=r_cut,
    E_max=E_max,
    all_species=all_species,
    le_type=le_type,
    factor=factor,
    factor2=0.0,
    cost_trade_off=False,
    fixed_stoichiometry=False,
    is_trace=False,
    n_trace=-1,
    device=device
)

accuracy_dict = le_ace.train(
    train_structures=train_structures,
    test_structures=test_structures,
    do_gradients=do_gradients,
    opt_target_name=opt_target_name,
    target_key=target_key,
    force_weight=force_weight,
    batch_size=batch_size
)

print("Validation RMSE energies:", accuracy_dict["validation RMSE energies"])
print("Validation MAE energies:", accuracy_dict["validation MAE energies"])
if do_gradients:
    print("Validation RMSE forces:", accuracy_dict["validation RMSE forces"])
    print("Validation MAE forces:", accuracy_dict["validation MAE forces"])

print("Test RMSE energies:", accuracy_dict["test RMSE energies"])
print("Test MAE energies:", accuracy_dict["test MAE energies"])
if do_gradients:
    print("Test RMSE forces:", accuracy_dict["test RMSE forces"])
    print("Test MAE forces:", accuracy_dict["test MAE forces"])


"""
# Semi-fast evaluator:
le_ace_predictor = le_ace.get_fast_evaluator()

import time
n_test = len(test_structures)
start_time = time.time()
for test_structure in test_structures:
    le_ace.predict([test_structure], do_positions_grad=True)
finish_time = time.time()
print(f"Evaluation took {(finish_time-start_time)/n_test} per structure")
"""
