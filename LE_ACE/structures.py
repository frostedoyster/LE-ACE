import rascaline.torch

def transform_structures(structures, positions_requires_grad=False, cells_requires_grad=False):
    return rascaline.torch.systems_to_torch(structures, positions_requires_grad, cells_requires_grad)
