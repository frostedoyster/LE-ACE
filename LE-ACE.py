import torch
import numpy as np
from datetime import datetime
import ase
from ase import io

from dataset_processing import get_dataset_slices
from error_measures import get_rmse, get_mae

from composition import get_composition_features
from lexicographic_multiplicities import apply_multiplicities
from spherical_bessel_zeros import Jn_zeros
from sum_like_atoms import sum_like_atoms

from LE_expansion import get_LE_expansion, write_spline
from LE_iterations import LEIterator
from LE_invariants import LEInvariantCalculator
from LE_regularization import get_LE_regularization

torch.set_default_dtype(torch.float64)
# torch.manual_seed(1234)
RANDOM_SEED = 304
np.random.seed(RANDOM_SEED)
print(f"Random seed: {RANDOM_SEED}", flush = True)

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"Calculating features on {device}")
BATCH_SIZE = 20

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCALMOL = 627.509608030593
KCAL_MOL_TO_MEV = 0.0433641153087705*1000.0

DATASET_PATH = "datasets/rmd17/toluene2.extxyz" # 'datasets/methane.extxyz'
print("Dataset path: " + DATASET_PATH)
TARGET_KEY = "energy"
CONVERSION_FACTOR = KCAL_MOL_TO_MEV  # HARTREE_TO_KCALMOL
FORCE_CONVERSION_FACTOR = CONVERSION_FACTOR  #/0.529177  # for methane

FORCE_WEIGHT = 1.0/30.0

n_test = 200
n_train = 50
do_gradients = True

print("toluene HARDER-ER", flush=True)

r_cut = 4.4  # Radius of the sphere
r_cut_rs = 5.5  # Radius for two-body potential
nu_max = 4
# E_max = [1e30, 500.0, 100.0, 70.0, 70.0]
if do_gradients:
    N = len(ase.io.read(DATASET_PATH, index = "0:1")[0].get_atomic_numbers())
    print(f"Number of atoms in each molecule: {N}")
    three_N_plus_one = 3*N+1
    n_train_effective = n_train*three_N_plus_one
else:
    n_train_effective = n_train
# E_max = [1e30, 1000.0, 200.0]
E_max = [1e30, 3.0*n_train_effective, 14.0*n_train_effective**(1.0/2.0), 24.0*n_train_effective**(1.0/3.0), 38.0*n_train_effective**(1.0/4.0)]
print(E_max)

if not np.all(np.array(E_max[:-1]) >= np.array(E_max[1:])): print("LE WARNING: max eigenvalues not in descending order")
assert len(E_max) == nu_max + 1 

# Decrease nu_max if LE threshold is too low:
for iota in range(1, nu_max+1):
    if E_max[iota] < iota*np.pi**2:
        print(f"Decreasing nu_max to {iota-1}")
        nu_max = iota-1
        break

""" test_slice = str(0) + ":" + str(n_test)
train_slice = str(n_test) + ":" + str(n_test+n_train) """
'''
train_slice = str(0) + ":" + str(n_train)
test_slice = str(n_train) + ":" + str(n_test+n_train)
'''

# rMD17
train_slice = str(0) + ":" + str(n_train)
test_slice = str(0) + ":" + str(n_test)

train_structures, test_structures = get_dataset_slices(DATASET_PATH, train_slice, test_slice)

train_energies = torch.tensor([structure.info[TARGET_KEY] for structure in train_structures])*CONVERSION_FACTOR
test_energies = torch.tensor([structure.info[TARGET_KEY] for structure in test_structures])*CONVERSION_FACTOR

if do_gradients:
    train_forces = torch.tensor(np.concatenate([structure.get_forces() for structure in train_structures], axis = 0))*FORCE_CONVERSION_FACTOR*FORCE_WEIGHT
    test_forces = torch.tensor(np.concatenate([structure.get_forces() for structure in test_structures], axis = 0))*FORCE_CONVERSION_FACTOR*FORCE_WEIGHT

l_big = 50
n_big = 50

z_ln = Jn_zeros(l_big, n_big)  # Spherical Bessel zeros
z_nl = z_ln.T

E_nl = z_nl**2

all_species = np.sort(np.unique(np.concatenate([train_structure.numbers for train_structure in train_structures] + [test_structure.numbers for test_structure in test_structures])))
print(f"All species: {all_species}")

n_max_rs = np.where(E_nl[:, 0] <= E_max[1])[0][-1] + 1

date_time = datetime.now()
date_time = date_time.strftime("%m-%d-%Y-%H-%M-%S")
rs_spline_path = "splines/splines-rs-" + date_time + ".txt"
write_spline(r_cut_rs, n_max_rs, 0, rs_spline_path)

if nu_max > 1:
    n_max = np.where(E_nl[:, 0] <= E_max[2])[0][-1] + 1
    l_max = np.where(E_nl[0, :] <= E_max[2])[0][-1]
    print(n_max, l_max)

    spline_path = "splines/splines-" + date_time + ".txt"
    write_spline(r_cut, n_max, l_max, spline_path)

    dummy_spherical_expansion = get_LE_expansion(train_structures[:2], spline_path, E_nl, E_max[2], r_cut, all_species, do_gradients=do_gradients, device=device)

    # anl counter:
    a_max = len(all_species)
    n_max_l = []
    a_i = all_species[0]
    for l in range(l_max+1):
        block = dummy_spherical_expansion.block(lam=l, a_i=a_i)
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

def get_LE_invariants(structures):

    print("Calculating composition features", flush = True)
    comp = get_composition_features(structures, all_species)
    print("Composition features done", flush = True)

    rs = get_LE_expansion(structures, rs_spline_path, E_nl, E_max[1], r_cut_rs, all_species, rs=True, do_gradients=do_gradients)
    if nu_max > 1: spherical_expansion = get_LE_expansion(structures, spline_path, E_nl, E_max[2], r_cut, all_species, do_gradients=do_gradients, device=device)

    invariants = [rs]

    if nu_max > 1: equivariants_nu_minus_one = spherical_expansion

    for nu in range(2, nu_max+1):

        print(f"Calculating nu = {nu} invariants", flush = True)
        invariants_nu = invariant_calculator(equivariants_nu_minus_one, spherical_expansion, E_max[nu])
        invariants_nu = apply_multiplicities(invariants_nu, combined_anl)
        invariants.append(invariants_nu)

        if nu == nu_max: break  # equivariants for nu_max wouldn't be used

        print(f"Calculating nu = {nu} equivariants", flush = True)
        equivariants_nu = equivariant_calculator(equivariants_nu_minus_one, spherical_expansion, E_max[nu+1])
        equivariants_nu_minus_one = equivariants_nu

    X, dX, LE_reg = sum_like_atoms(comp, invariants, all_species, E_nl)
    return X, dX, LE_reg


# Batch training and test set:

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
    X_train = torch.concat(X_train + dX_train, dim = 0)
else:
    X_train = torch.concat(X_train, dim = 0)

X_test = []
if do_gradients: dX_test = []
for i_batch, batch in enumerate(test_structures):
    print(f"DOING TEST BATCH {i_batch+1} out of {len(test_structures)}")
    X, dX, LE_reg = get_LE_invariants(batch)
    X_test.append(X)
    if do_gradients: dX_test.append(-FORCE_WEIGHT*dX.reshape(dX.shape[0]*3, dX.shape[2]))  # note the minus sign

if do_gradients:
    X_test = torch.concat(X_test + dX_test, dim = 0)
else:
    X_test = torch.concat(X_test, dim = 0)

print("Features done", flush = True)

'''
print("Normalizing invariants...", flush = True)

for j in range(X_train.shape[1])[5:]:   # HARDCODED SHIT... 5 is the number of nu = 0 stuff
    if j % 1000 == 0: print(X_train[:, j])
    mean = torch.mean(X_train[:, j])
    if j % 1000 == 0: print(mean)
    X_train[:, j] -= 0.0
    X_test[:, j] -= 0.0
    stddev = torch.sqrt(torch.mean(X_train[:, j]**2))
    if j % 1000 == 0: print(stddev)
    if stddev == 0.0: continue   # Features that are zero either geometrically or due to generalized CG degeneracies
    X_train[:, j] /= stddev
    X_test[:, j] /= stddev

print("Normalization done", flush = True)
'''

# validation_cycle = ValidationCycleLinear(alpha_exp_initial_guess = -5.0)

print("Beginning hyperparameter optimization")

""" def validation_loss_for_global_optimization(x):

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
    rmses.append(loss) """

if do_gradients:
    train_targets = torch.concat([train_energies, train_forces.reshape((-1,))])
    test_targets = torch.concat([test_energies, test_forces.reshape((-1,))])
else:
    train_targets = train_energies
    test_targets = test_energies

symm = X_train.T @ X_train
vec = X_train.T @ train_targets
opt_target = []
alpha_list = np.linspace(-20, 0, 41)
n_feat = X_train.shape[1]
print("Number of features: ", n_feat)

for alpha in alpha_list:
    # torch.cholesky_solve(X_train.T @ train_targets.reshape(-1, 1), symm + 10.0**alpha * torch.eye(n_feat))
    for i in range(n_feat):
        symm[i, i] += 10**alpha*LE_reg[i]

    try:
        # c = torch.linalg.lstsq(X_train, train_targets, rcond = 10**alpha, driver = "gelsd").solution
        c = torch.linalg.solve(symm, vec)
    except Exception as e:
        print(e)
        opt_target.append(1e30)
        continue
    train_predictions = X_train @ c
    test_predictions = X_test @ c
    print(alpha, get_rmse(train_predictions[:n_train], train_targets[:n_train]).item(), get_rmse(test_predictions[:n_test], test_targets[:n_test]).item(), get_mae(test_predictions[:n_test], test_targets[:n_test]).item(), get_rmse(train_predictions[n_train:], train_targets[n_train:]).item()/FORCE_WEIGHT, get_rmse(test_predictions[n_test:], test_targets[n_test:]).item()/FORCE_WEIGHT, get_mae(test_predictions[n_test:], test_targets[n_test:]).item()/FORCE_WEIGHT, flush=True)
    # rmses.append(get_mae(test_predictions, test_energies).item())
    opt_target.append(get_mae(test_predictions[:n_test], test_targets[:n_test]).item())

    for i in range(n_feat):
        symm[i, i] -= 10.0**alpha*LE_reg[i]

best_alpha = alpha_list[np.argmin(opt_target)]
print(best_alpha, np.min(opt_target))

for i in range(n_feat):
    symm[i, i] += 10**best_alpha*LE_reg[i]
c = torch.linalg.solve(symm, vec)
test_predictions = X_test @ c
print("n_train:", n_train, "n_features:", n_feat)
print(f"Test set RMSE (E): {get_rmse(test_predictions[:n_test], test_targets[:n_test]).item()} [MAE (E): {get_mae(test_predictions[:n_test], test_targets[:n_test]).item()}], RMSE (F): {get_rmse(test_predictions[n_test:], test_targets[n_test:]).item()/FORCE_WEIGHT} [MAE (F): {get_mae(test_predictions[n_test:], test_targets[n_test:]).item()/FORCE_WEIGHT}]")

'''
print((X_test[n_train:]@c)[:100]/FORCE_WEIGHT)
print(test_targets[n_train:][:100]/FORCE_WEIGHT)
'''
