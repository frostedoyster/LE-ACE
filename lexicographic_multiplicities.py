import torch
import math
import numpy as np
from equistore import TensorBlock, TensorMap

def apply_multiplicities(old_map: TensorMap, unified_anl) -> TensorMap:
    # CAREFUL. ASSUMES ALL ELEMENT CENTER BLOCKS ARE THE SAME

    nu = len(old_map.block(0).properties.names)//4
    block = old_map.block(0)
    
    multiplicities = []
    for i_feature in range(block.values.shape[-1]):
        I = []
        for iota in range(1, nu+1):
            index = unified_anl[(block.properties["a"+str(iota)][i_feature], block.properties["n"+str(iota)][i_feature], block.properties["l"+str(iota)][i_feature])]
            I.append(index)
        unique, counts = np.unique(I, return_counts=True)
        multiplicity = math.factorial(nu)
        for count in counts:
            multiplicity = multiplicity // math.factorial(count)
        # multiplicity = np.sqrt(multiplicity)
        multiplicities.append(multiplicity)
    multiplicities = torch.tensor(multiplicities, device=old_map.block(0).values.device)
    # print(multiplicities)

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