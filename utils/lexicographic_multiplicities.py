import torch
import math
import numpy as np
import scipy as sp
from equistore import TensorBlock, TensorMap

def apply_multiplicities(old_map: TensorMap, unified_anl) -> TensorMap:
    # Assumes all center elements are truncated in the same way

    nu = len(old_map.block(0).properties.names)//4
    block = old_map.block(0)
    
    """
    multiplicities = []
    for i_feature in range(block.values.shape[-1]):
        I = []
        for iota in range(1, nu+1):
            index = unified_anl[(block.properties["a"+str(iota)][i_feature], block.properties["n"+str(iota)][i_feature], block.properties["l"+str(iota)][i_feature])]
            I.append(index)
        unique, counts = np.unique(I, return_counts=True)
        multiplicity = math.factorial(nu)
        for count in counts:
            multiplicity = multiplicity/math.factorial(count)
        multiplicity = np.sqrt(multiplicity)
        multiplicities.append(multiplicity)
    multiplicities = torch.tensor(multiplicities, device=old_map.block(0).values.device)
    # print(multiplicities)
    """

    anl_indices = []
    for iota in range(1, nu+1):
        anl_indices.append([block.properties["a"+str(iota)], block.properties["n"+str(iota)], block.properties["l"+str(iota)]]) 
    anl_indices = np.array(anl_indices)
    anl_indices = anl_indices.swapaxes(0, 2).swapaxes(1, 2)
    denominators = []
    for i in range(anl_indices.shape[0]):
        _, count = np.unique(anl_indices[i], axis=0, return_counts=True)
        denominator = np.prod(sp.special.gamma(count + 1))
        denominators.append(denominator)
    multiplicities = math.factorial(nu)/np.array(denominators)
    multiplicities = np.sqrt(multiplicities)
    multiplicities = torch.tensor(multiplicities, device=old_map.block(0).values.device)

    new_blocks = []
    for _, block in old_map:
        new_block = TensorBlock(
            values=block.values*multiplicities,
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )
        if block.has_gradient("positions"): new_block.add_gradient(
            "positions",
            data = block.gradient("positions").data*multiplicities, 
            samples = block.gradient("positions").samples, 
            components = block.gradient("positions").components,
        )
        new_blocks.append(new_block)
    return TensorMap(
            keys = old_map.keys,
            blocks = new_blocks
            )
