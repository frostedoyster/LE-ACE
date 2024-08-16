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
E_max = [-1, 2000.0, 350.0, 220.0, 50.0]
device = "cuda" if torch.cuda.is_available() else "cpu"

# Training options:
do_gradients = True
opt_target_name = "rmse"
target_key = "energy"
force_weight = 0.1
batch_size = 2

all_structures = ase.io.read("datasets/C_dataset.xyz", ":")
np.random.shuffle(all_structures)

train_structures = all_structures[1000:2000]
test_structures = all_structures[:1000]

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


test_structures_by_category = {
    'sp2 bonded': [],
    'sp3 bonded': [],
    'amorphous/liquid': [],
    'general bulk': [],
    'general clusters': [],
}
for test_structure in test_structures:
    category = test_structure.info['category']
    test_structures_by_category[category].append(test_structure)


for category, structures in test_structures_by_category.items():

    print()
    print("Results for category", category)

    sse_e = 0.0
    sse_f = 0.0
    n_e = 0
    n_f = 0

    for structure in structures:
        predictions = le_ace.predict([structure], do_positions_grad=True)
        e, f = predictions["values"], -predictions["positions gradient"]
        true_e = structure.info["energy"]
        true_f = structure.arrays["forces"]
        sse_e += (e.item()/len(structure)-true_e/len(structure))**2
        n_e += 1
        sse_f += np.sum((f.clone().detach().cpu().numpy() - true_f)**2)
        n_f += 3* len(structure)

    print("Energy RMSE", np.sqrt(sse_e/n_e))
    print("Force RMSE", np.sqrt(sse_f/n_f))
