import numpy as np
import torch

class ValidationCycle(torch.nn.Module):
    # Evaluates the model on the validation set so that derivatives 
    # of an arbitrary loss with respect to the continuous
    # hyperparameters can be used to minimize the validation loss.

    def __init__(self, alpha_exp_initial_guess):
        super().__init__()

        # Kernel regularization:
        self.sigma_exponent = torch.nn.Parameter(
            torch.tensor([alpha_exp_initial_guess], dtype = torch.get_default_dtype())
            )

    def forward(self, K_train, y_train, K_val):
        sigma = torch.exp(self.sigma_exponent*np.log(10.0))
        n_train = K_train.shape[0] 
        c = torch.linalg.solve(
        K_train +
        sigma * torch.eye(n_train)  # regularization
        , 
        y_train)
        y_val_predictions = K_val @ c

        return y_val_predictions


class ValidationCycleLinear(torch.nn.Module):
    # Evaluates the model on the validation set so that derivatives 
    # of an arbitrary loss with respect to the continuous
    # hyperparameters can be used to minimize the validation loss.

    def __init__(self, alpha_exp_initial_guess):
        super().__init__()

        # Kernel regularization:
        self.sigma_exponent = torch.nn.Parameter(
            torch.tensor([alpha_exp_initial_guess], dtype = torch.get_default_dtype())
            )

    def forward(self, X_train, y_train, X_val):
        sigma = torch.exp(self.sigma_exponent*np.log(10.0))
        n_feat = X_train.shape[1] 
        c = torch.linalg.solve(
        X_train.T @ X_train +
        sigma * torch.eye(n_feat),  # regularization
        X_train.T @ y_train)
        y_val_predictions = X_val @ c

        return y_val_predictions