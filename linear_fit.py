import numpy as np
import torch
import argparse

from utils.error_measures import get_rmse, get_mae

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

    symm = X_train.T @ X_train
    vec = X_train.T @ train_targets
    alpha_list = np.linspace(-12.5, -2.5, 21)

    n_feat = X_train.shape[1]
    print("Number of features: ", n_feat)

    best_opt_target = 1e30
    for beta in [-2.0, -1.0, 0.0, 1.0, 2.0]:  # reproduce
        print("beta=", beta)

        LE_reg = torch.load(LE_reg_path)
        nu_max = len(LE_reg) - 1
        for nu in range(nu_max+1):
            LE_reg[nu] *= np.exp(beta*nu)
        LE_reg = torch.concatenate(LE_reg)

        for alpha in alpha_list-beta*1.5:
            #X_train[-1, :] = torch.sqrt(10**alpha*LE_reg)
            for i in range(n_feat):
                symm[i, i] += 10**alpha*LE_reg[i]

            try:
                # c = torch.linalg.lstsq(X_train, train_targets, driver = "gelsd", rcond = 1e-8).solution
                c = torch.linalg.solve(symm, vec)
                # c, info = cg(symm.numpy(force=True), vec.numpy(force=True), atol=1e-10, tol=1e-12)
                # c = torch.tensor(c)
                """if info != 0:
                    print(info)
                    opt_target.append(1e30)
                    for i in range(n_feat):
                        symm[i, i] -= 10.0**alpha*LE_reg[i]
                    continue"""
                # c = mkl64.dposv(symm, vec)
            except Exception as e:
                print(e)
                opt_target = 1e30
                for i in range(n_feat):
                    symm[i, i] -= 10.0**alpha*LE_reg[i]
                continue
            train_predictions = X_train @ c
            test_predictions = X_test @ c
            # print("Residual:", torch.sqrt(torch.sum((vec-symm@c)**2)))

            print(alpha, get_rmse(train_predictions[:n_train], train_targets[:n_train]).item(), get_rmse(test_predictions[:n_test], test_targets[:n_test]).item(), get_mae(test_predictions[:n_test], test_targets[:n_test]).item(), get_rmse(train_predictions[n_train:], train_targets[n_train:]).item()/FORCE_WEIGHT, get_rmse(test_predictions[n_test:], test_targets[n_test:]).item()/FORCE_WEIGHT, get_mae(test_predictions[n_test:], test_targets[n_test:]).item()/FORCE_WEIGHT)
            if opt_target_name == "mae":
                if do_gradients:
                    opt_target = get_mae(test_predictions[n_test:], test_targets[n_test:]).item()/FORCE_WEIGHT
                else:
                    opt_target = get_mae(test_predictions[:n_test], test_targets[:n_test]).item()
            else:
                if do_gradients:
                    opt_target = get_rmse(test_predictions[n_test:], test_targets[n_test:]).item()/FORCE_WEIGHT
                else:    
                    opt_target = get_rmse(test_predictions[:n_test], test_targets[:n_test]).item()

            if opt_target < best_opt_target:
                best_opt_target = opt_target
                best_alpha = alpha
                best_beta = beta

            for i in range(n_feat):
                symm[i, i] -= 10.0**alpha*LE_reg[i]

    print("Best parameters:", best_alpha, best_beta)
    if best_beta == -2.0 or best_beta == 2.0:
        print("WARNING: hit grid search boundary")

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
