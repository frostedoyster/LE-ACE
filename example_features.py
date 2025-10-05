import numpy as np
import torch
import ase

from LE_ACE import LE_ACE

torch.set_default_dtype(torch.float64)

# Model hyperparameters
r_cut = 6.0
r_cut_rs = 6.0
le_type = "physical"
factor = 2.0
E_max = [-1, 1500.0, 200.0, 100.0, 75.0]
device = "cuda" if torch.cuda.is_available() else "cpu"

structures = [
    ase.Atoms("H2O", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=[False, False, False]),
    ase.Atoms("NH3", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=[False, False, False])
]
all_species = np.sort(np.unique(np.concatenate([structure.numbers for structure in structures])))

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
    is_trace=True,
    n_trace=5,
    device=device
)

features = le_ace.compute_features(structures)
