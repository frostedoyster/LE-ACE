import torch
import numpy as np

from equistore import Labels, TensorBlock, TensorMap

from dataset_processing import get_dataset_slices
from error_measures import get_sse, get_rmse, get_mae
from validation import ValidationCycleLinear

from LE_expansion import get_LE_expansion, write_spline
from spherical_bessel_zeros import Jn_zeros
from LE_iterations import LEIterator
from LE_invariants import LEInvariantCalculator

# torch.set_default_dtype(torch.float64)
# torch.manual_seed(1234)
RANDOM_SEED = 1000
np.random.seed(RANDOM_SEED)
print(f"Random seed: {RANDOM_SEED}", flush = True)

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCALMOL = 627.5

DATASET_PATH = 'datasets/random-ch4-10k.extxyz'
TARGET_KEY = "energy"
CONVERSION_FACTOR = HARTREE_TO_KCALMOL

n_test = 1000
n_train = 1000
do_gradients = False

r_cut = 6.0  # Radius of the sphere
nu_max = 1
E_max = [0.0, 0.4*n_train] # 7.0*n_train**(1.0/2.0), 14.0*n_train**(1.0/3.0), 22.0*n_train**(1.0/4.0)]
assert len(E_max) == nu_max + 1 

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

train_structures, test_structures = get_dataset_slices(DATASET_PATH, train_slice, test_slice)

l_big = 26
n_big = 26

z_ln = Jn_zeros(l_big, n_big)  # Spherical Bessel zeros
z_nl = z_ln.T

E_nl = z_nl**2


n_max_rs = np.where(E_nl[:, 0] <= E_max[1])[0][-1] + 1
write_spline(r_cut, n_max_rs, 0, "splines_rs.txt")
train_rs = get_LE_expansion(train_structures, "splines_rs.txt", E_nl, E_max[1], r_cut, rs=True, do_gradients=do_gradients)
test_rs = get_LE_expansion(test_structures, "splines_rs.txt", E_nl, E_max[1], r_cut, rs=True, do_gradients=do_gradients)
all_species = np.unique(np.concatenate([train_rs.keys["a_i"], test_rs.keys["a_i"]]))

if nu_max > 1:
    n_max = np.where(E_nl[:, 0] <= E_max[2])[0][-1] + 1
    l_max = np.where(E_nl[0, :] <= E_max[2])[0][-1]
    print(n_max, l_max)

    write_spline(r_cut, n_max, l_max, "splines.txt")
    spherical_expansion_train = get_LE_expansion(train_structures, "splines.txt", E_nl, E_max[2], r_cut, do_gradients=do_gradients)
    spherical_expansion_test = get_LE_expansion(test_structures, "splines.txt", E_nl, E_max[2], r_cut, do_gradients=do_gradients)

    # anl counter:
    a_max = len(all_species)
    n_max_l = []
    a_i = all_species[0]
    for l in range(l_max+1):
        block = spherical_expansion_train.block(lam=l, a_i=a_i)
        n = block.properties["n1"]
        n_max_l.append(np.max(n)+1)
    combined_anl = {}
    anl_counter = 0
    for a in range(a_max):
        for l in range(l_max+1):
            for n in range(n_max_l[l]):
                combined_anl[(a, n, l,)] = anl_counter
                anl_counter += 1

    invariant_calculator = LEInvariantCalculator(E_nl, combined_anl, all_species)
    equivariant_calculator = LEIterator(E_nl, combined_anl, all_species)

print("Calculating composition features", flush = True)
train_comp = get_composition_features(train_structures, all_species)
test_comp = get_composition_features(test_structures, all_species)
print("Composition features done", flush = True)

## NEED FUNCTION TO SUM COMPOSITION IN CASE IT'S A MD-LIKE DATASET

train_invariants = [train_rs]
test_invariants = [test_rs]

if nu_max > 1:
    train_equivariants_nu_minus_one = spherical_expansion_train
    test_equivariants_nu_minus_one = spherical_expansion_test

for nu in range(2, nu_max+1):

    print(f"Calculating nu = {nu} invariants")
    train_invariants_nu = invariant_calculator(train_equivariants_nu_minus_one, spherical_expansion_train, E_max[nu])
    test_invariants_nu = invariant_calculator(test_equivariants_nu_minus_one, spherical_expansion_test, E_max[nu])

    train_invariants.append(train_invariants_nu)
    test_invariants.append(test_invariants_nu)

    if nu == nu_max: break  # equivariants for nu_max wouldn't be used

    print(f"Calculating nu = {nu} equivariants")
    train_equivariants_nu = equivariant_calculator(train_equivariants_nu_minus_one, spherical_expansion_train, E_max[nu+1])
    test_equivariants_nu = equivariant_calculator(test_equivariants_nu_minus_one, spherical_expansion_test, E_max[nu+1])

    train_equivariants_nu_minus_one = train_equivariants_nu
    test_equivariants_nu_minus_one = test_equivariants_nu

def sum_over_like_atoms(comp, invariants, species):

    n_structures = len(comp.samples["structure"])
    n_features = [invariants_nu.block(0).values.shape[1] for invariants_nu in invariants]

    for nu_minus_one in range(len(invariants)):
        print(f"nu = {nu_minus_one+1}:", n_features[nu_minus_one]*len(species))

    features = []
  
    for nu_minus_one in range(len(invariants)):
        for center_species in species:
            features_current_center_species = torch.zeros((n_structures, n_features[nu_minus_one]))

            # if center_species == 1: continue  # UNCOMMENT FOR METHANE DATASET C-ONLY VERSION
            print(f"     Calculating structure features for center species {center_species}", flush = True)
            try:
                structures = invariants[nu_minus_one].block(a_i=center_species).samples["structure"]
            except ValueError:
                print("WARNING: This set does not contain the above center species")
                continue

            len_samples = structures.shape[0]

            center_features = invariants[nu_minus_one].block(a_i=center_species).values

            for i in range(len_samples):
                features_current_center_species[structures[i], :] += center_features[i, :]

            features.append(features_current_center_species)

    comp = comp.values

    comp = torch.ones(features[0].shape[0], 1)

    X = torch.concat([comp] + features, dim = -1)
    return X

X_train = sum_over_like_atoms(train_comp, train_invariants, all_species)
X_test = sum_over_like_atoms(test_comp, test_invariants, all_species)

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
alpha_list = np.linspace(-5, 0, 20)
for alpha in alpha_list:
    loss = validation_loss_for_global_optimization([alpha])
    print(alpha, loss)
    rmses.append(loss)

best_sigma = alpha_list[np.argmin(rmses)]
best_sigma = np.exp(best_sigma*np.log(10.0))

print(best_sigma, np.sqrt(np.min(rmses)/n_train))

c = torch.linalg.solve(
    X_train.T @ X_train +
    best_sigma * torch.eye(X_train.shape[1])  # regularization
    , X_train.T @ train_energies)

test_predictions = X_test @ c
print("n_train:", n_train, "n_features:", c.shape[0])
print(f"Test set RMSE: {get_rmse(test_predictions, test_energies).item()} [MAE: {get_mae(test_predictions, test_energies).item()}]")


import ase
import matplotlib.pyplot as plt
plot_structures = []
radii = np.linspace(0.1, 3.0, 100)
for r in radii:
    plot_structures.append(ase.Atoms("CH",  positions=[(0, 0, 0), (r, 0 ,0)]))
plot_structures.append(ase.Atoms("CH",  positions=[(0, 0, 0), (1.0, 0 ,0)]))
plot_rs = get_LE_expansion(plot_structures, "splines_rs.txt", E_nl, E_max[1], r_cut, rs=True, do_gradients=do_gradients)
plot_comp = get_composition_features(plot_structures, all_species)
X_plot = sum_over_like_atoms(plot_comp, [plot_rs], all_species)

plot_energies = X_plot @ c

plt.plot(np.concatenate([radii, np.array([10.0])]), plot_energies, label="CH_second_kind")
plt.xlim(0.1, 3.0)
plt.ylim(np.min(plot_energies.numpy()), np.max(plot_energies.numpy()[13:-1]))
plt.legend()
plt.savefig("pairCH_second.png")