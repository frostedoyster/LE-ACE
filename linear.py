import torch
import numpy as np
import scipy as sp
from scipy import optimize

from equistore import Labels, TensorBlock, TensorMap
from rascaline import SoapRadialSpectrum, SoapPowerSpectrum

from dataset_processing import get_dataset_slice
from error_measures import get_sse, get_rmse, get_mae
from validation import ValidationCycleLinear

# torch.set_default_dtype(torch.float64)
# torch.manual_seed(1234)
RANDOM_SEED = 1000
np.random.seed(RANDOM_SEED)
print(f"Random seed: {RANDOM_SEED}", flush = True)

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCALMOL = 627.5

DATASET_PATH = 'datasets/qm9.xyz'
TARGET_KEY = "U0"
CONVERSION_FACTOR = HARTREE_TO_KCALMOL

n_test = 500
n_train = 100

n_validation_splits = 10
assert n_train % n_validation_splits == 0
n_validation = n_train // n_validation_splits
n_train_sub = n_train - n_validation

test_slice = str(0) + ":" + str(n_test)
train_slice = str(n_test) + ":" + str(n_test+n_train)

# Spherical expansion and composition

def get_composition_features(frames, all_species):
    species_dict = {s: i for i, s in enumerate(all_species)}
    data = torch.zeros((len(frames), len(species_dict)))
    for i, f in enumerate(frames):
        for s in f.numbers:
            data[i, species_dict[s]] += 1
    properties = Labels(
        names=["atomic_number"],
        values=np.array(list(species_dict.keys()), dtype=np.int32).reshape(
            -1, 1
        ),
    )

    frames_i = np.arange(len(frames), dtype=np.int32).reshape(-1, 1)
    samples = Labels(names=["structure"], values=frames_i)

    block = TensorBlock(
        values=data, samples=samples, components=[], properties=properties
    )
    composition = TensorMap(Labels.single(), blocks=[block])
    return composition.block()

hypers_rs = {
    "cutoff": 4.5,
    "max_radial": 12,
    "atomic_gaussian_width": 0.05,
    "center_atom_weight": 0.0,
    "radial_basis": {"Gto": {"spline_accuracy": 1e-8}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "radial_scaling":  {"Willatt2018": { "scale": 2.0, "rate": 2.0, "exponent": 6}},
}

hypers_ps = {
    "cutoff": 4.5,
    "max_radial": 6,
    "max_angular": 4,
    "atomic_gaussian_width": 0.05,
    "center_atom_weight": 0.0,
    "radial_basis": {"Gto": {"spline_accuracy": 1e-8}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "radial_scaling":  {"Willatt2018": { "scale": 2.0, "rate": 2.0, "exponent": 6}},
}

calculator_ps = SoapPowerSpectrum(**hypers_ps)
calculator_rs = SoapRadialSpectrum(**hypers_rs)

train_structures = get_dataset_slice(DATASET_PATH, train_slice)
test_structures = get_dataset_slice(DATASET_PATH, test_slice)

def move_to_torch(rust_map: TensorMap) -> TensorMap:
    torch_blocks = []
    for _, block in rust_map:
        torch_block = TensorBlock(
            values=torch.tensor(block.values).to(dtype=torch.get_default_dtype()),
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )
        torch_blocks.append(torch_block)
    return TensorMap(
            keys = rust_map.keys,
            blocks = torch_blocks
            )

print("Calculating invariants", flush = True)

train_rs = calculator_rs.compute(train_structures)
train_rs = move_to_torch(train_rs)

test_rs = calculator_rs.compute(test_structures)
test_rs = move_to_torch(test_rs)

train_ps = calculator_ps.compute(train_structures)
train_ps = move_to_torch(train_ps)

test_ps = calculator_ps.compute(test_structures)
test_ps = move_to_torch(test_ps)

all_species = np.unique(np.concatenate([train_ps.keys["species_center"], test_ps.keys["species_center"]]))

print("Calculating composition features", flush = True)
train_comp = get_composition_features(train_structures, all_species)
test_comp = get_composition_features(test_structures, all_species)
print("Composition features done", flush = True)

def sum_over_like_atoms(comp, rs, ps, species):

    all_neighbor_species = Labels(
            names=["species_neighbor"],
            values=np.array(species, dtype=np.int32).reshape(-1, 1),
        )

    all_neighbor_species_1 = Labels(
            names=["species_neighbor_1"],
            values=np.array(species, dtype=np.int32).reshape(-1, 1),
        )

    all_neighbor_species_2 = Labels(
            names=["species_neighbor_2"],
            values=np.array(species, dtype=np.int32).reshape(-1, 1),
        )

    rs.keys_to_properties(all_neighbor_species)
    ps.keys_to_properties(all_neighbor_species_1)
    ps.keys_to_properties(all_neighbor_species_2)

    n_structures = len(comp.samples["structure"])
    n_rs_features = rs.block(0).values.shape[1]
    n_ps_features = ps.block(0).values.shape[1]

    rs_features = []
    ps_features = []
  
    for center_species in species:
        rs_features_current_center_species = torch.zeros((n_structures, n_rs_features))
        ps_features_current_center_species = torch.zeros((n_structures, n_ps_features))

        # if center_species == 1: continue  # UNCOMMENT FOR METHANE DATASET C-ONLY VERSION
        print(f"     Calculating structure features for center species {center_species}", flush = True)
        try:
            structures_rs = rs.block(species_center=center_species).samples["structure"]
            structures_ps = ps.block(species_center=center_species).samples["structure"]
        except ValueError:
            print("This set does not contain the above center species")
            exit()

        len_rs = structures_rs.shape[0]
        len_ps = structures_ps.shape[0]

        print(len_rs, len_ps)

        center_features_rs = rs.block(species_center=center_species).values
        center_features_ps = ps.block(species_center=center_species).values

        for i in range(len_rs):
            rs_features_current_center_species[structures_rs[i], :] += center_features_rs[i, :]

        for i in range(len_ps):
            ps_features_current_center_species[structures_ps[i], :] += center_features_ps[i, :]

        rs_features.append(rs_features_current_center_species)
        ps_features.append(ps_features_current_center_species)

    rs_features = torch.concat(rs_features, dim = -1)
    ps_features = torch.concat(ps_features, dim = -1)

    comp = comp.values
    print(comp.shape, rs_features.shape, ps_features.shape)
    X = torch.concat([comp, rs_features, ps_features], dim = -1)
    return X

X_train = sum_over_like_atoms(train_comp, train_rs, train_ps, all_species)
X_test = sum_over_like_atoms(test_comp, test_rs, test_ps, all_species)

print("Features done", flush = True)

'''
L2_mean = get_L2_mean(train_coefs)
#print(L2_mean)
for key in train_coefs.keys():
    train_coefs[key] /= np.sqrt(L2_mean)
    test_coefs[key] /= np.sqrt(L2_mean)
'''

train_energies = [structure.info[TARGET_KEY] for structure in train_structures]
train_energies = torch.tensor(train_energies, dtype = torch.get_default_dtype()) * CONVERSION_FACTOR

test_energies = [structure.info[TARGET_KEY] for structure in test_structures]
test_energies = torch.tensor(test_energies, dtype = torch.get_default_dtype()) * CONVERSION_FACTOR

# nu = 0 contribution

'''
X_train = X_train[:, :5]
X_test = X_test[:, :5]
'''
'''
symm = X_train.T @ X_train

for alpha in np.linspace(-10, 0, 20):
    alpha = 10.0**alpha
    try:
        c = torch.linalg.solve(symm + alpha*torch.eye(X_train.shape[1]), X_train.T @ train_energies)
    except Exception as e:
        print(alpha, e)
        continue
    print(alpha, get_rmse(train_energies, X_train @ c), get_rmse(test_energies, X_test @ c))
'''

validation_cycle = ValidationCycleLinear(alpha_exp_initial_guess = -5.0)

print("Beginning hyperparameter optimization")

def validation_loss_for_global_optimization(x):

    validation_cycle.sigma_exponent = torch.nn.Parameter(
            torch.tensor(x[-1], dtype = torch.get_default_dtype())
        )

    validation_loss = 0.0
    for i_validation_split in range(n_validation_splits):
        index_validation_start = i_validation_split*n_validation
        index_validation_stop = index_validation_start + n_validation

        X_train_sub = torch.empty((n_train_sub, X_train.shape[1]))
        X_train_sub[:index_validation_start, :] = X_train[:index_validation_start, :]
        if i_validation_split != n_validation_splits - 1:
            X_train_sub[index_validation_start:, :] = X_train[index_validation_stop:, :]
        y_train_sub = train_energies[:index_validation_start]
        if i_validation_split != n_validation_splits - 1:
            y_train_sub = torch.concat([y_train_sub, train_energies[index_validation_stop:]])

        X_validation = X_train[index_validation_start:index_validation_stop, :]
        y_validation = train_energies[index_validation_start:index_validation_stop] 

        with torch.no_grad():
            validation_predictions = validation_cycle(X_train_sub, y_train_sub, X_validation)
            validation_loss += get_sse(validation_predictions, y_validation).item()
    '''
    with open("log.txt", "a") as out:
        out.write(str(np.sqrt(validation_loss/n_train)) + "\n")
        out.flush()
    '''

    return validation_loss

symm = X_train.T @ X_train
rmses = []
for alpha in np.linspace(-10, 0, 20):
    loss = validation_loss_for_global_optimization([alpha])
    print(alpha, loss)
    rmses.append(loss)

best_sigma = np.linspace(-10, 0, 20)[np.argmin(rmses)]
best_sigma = np.exp(best_sigma*np.log(10.0))

print(best_sigma, np.sqrt(np.min(rmses)/n_train))

print("WTF:", X_train.shape, X_test.shape)

c = torch.linalg.solve(
    X_train.T @ X_train +
    best_sigma * torch.eye(X_train.shape[1])  # regularization
    , X_train.T @ train_energies)

print(X_test.shape, c.shape)
test_predictions = X_test @ c
print(f"Test set RMSE (after kernel mixing): {get_rmse(test_predictions, test_energies).item()}")

print()
print("Final result (test MAE):")
print(n_train, get_mae(test_predictions, test_energies).item())