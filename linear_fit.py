import numpy as np
import torch
import argparse

from utils.solver import Solver
from utils.error_measures import get_rmse, get_mae, get_sse, get_sae

torch.set_default_dtype(torch.float64)

def run_fit(X_train_path, X_test_path, y_train_path, y_test_path, LE_reg_path, molecule, identifier):

    print("DOING", molecule)
    print("Identifier:", identifier)

    n_train = 1000
    n_test = 1000
    FORCE_WEIGHT = 0.03
    opt_target_name = "mae"
    do_gradients = True

    X_train = torch.load(X_train_path)
    X_test = torch.load(X_test_path)
    train_targets = torch.load(y_train_path)
    test_targets = torch.load(y_test_path)
    LE_reg = torch.load(LE_reg_path)
    nu_max = len(LE_reg) - 1

    symm = X_train.T @ X_train
    vec = X_train.T @ train_targets
    alpha_list = np.linspace(-12.5, -2.5, 21)

    n_feat = X_train.shape[1]
    print("Number of features: ", n_feat)

    alpha_start = -10.0
    beta_start = 0.0

    solver = Solver(n_feat, LE_reg, alpha_start, beta_start, nu_max)
    optimizer = torch.optim.LBFGS(solver.parameters(), max_iter=5)

    loss_list = []
    alpha_list = []
    beta_list = []

    def closure():
        optimizer.zero_grad()

        c = solver(symm, vec)
        validation_predictions = X_validation @ c

        if opt_target_name == "mae":
            loss = get_sae(validation_predictions, validation_targets)
        else:
            loss = get_sse(validation_predictions, validation_targets)
        
        print(f"alpha={solver.alpha.item()} beta={solver.beta.item()} loss={loss.item()}")
        loss_list.append(loss.item())
        alpha_list.append(solver.alpha.item())
        beta_list.append(solver.beta.item())

        loss.backward()
        return loss

    n_cycles = 4
    for i_cycle in range(n_cycles):
        _ = optimizer.step(closure)
        print(f"Finished step {i_cycle+1} out of {n_cycles}")

    where_best_loss = np.argmin(np.nan_to_num(loss_list, nan=1e100))
    best_alpha = alpha_list[where_best_loss]
    best_beta = beta_list[where_best_loss]
    print("Best parameters:", best_alpha, best_beta)

    LE_reg = torch.load(LE_reg_path)
    for nu in range(nu_max+1):
        LE_reg[nu] *= np.exp(best_beta*nu)
    LE_reg = torch.concatenate(LE_reg)  

    for i in range(n_feat):
        symm[i, i] += 10**best_alpha*LE_reg[i]
    c = torch.linalg.solve(symm, vec)

    test_predictions = X_test @ c
    print("n_train:", n_train, "n_features:", n_feat)
    print(f"Test set RMSE (E): {get_rmse(test_predictions[:n_test], test_targets[:n_test]).item()} [MAE (E): {get_mae(test_predictions[:n_test], test_targets[:n_test]).item()}], RMSE (F): {get_rmse(test_predictions[n_test:], test_targets[n_test:]).item()/FORCE_WEIGHT} [MAE (F): {get_mae(test_predictions[n_test:], test_targets[n_test:]).item()/FORCE_WEIGHT}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="?"
    )

    parser.add_argument(
        "X_train_path",
        type=str,
    )
    parser.add_argument(
        "X_test_path",
        type=str,
    )
    parser.add_argument(
        "y_train_path",
        type=str,
    )
    parser.add_argument(
        "y_test_path",
        type=str,
    )
    parser.add_argument(
        "LE_reg_path",
        type=str,
    )
    parser.add_argument(
        "molecule",
        type=str,
    )
    parser.add_argument(
        "identifier",
        type=str,
    )

    args = parser.parse_args()
    run_fit(args.X_train_path, args.X_test_path, args.y_train_path, args.y_test_path, args.LE_reg_path, args.molecule, args.identifier)
