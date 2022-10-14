import numpy as np
import ase
from ase import io

def get_dataset_slices(dataset_path, train_slice, test_slice):
    
    if "methane" in dataset_path:
        train_structures = ase.io.read(dataset_path, index = train_slice)
        test_structures = ase.io.read(dataset_path, index = test_slice)
    else:  # QM7 and QM9 don't seem to be shuffled randomly 
        print("Reading dataset")
        all_structures = ase.io.read(dataset_path, index = ":")
        print(f"Shuffling and extracting from dataset (length: {len(all_structures)})")
        np.random.shuffle(all_structures)
        train_index_begin = int(train_slice.split(":")[0])
        train_index_end = int(train_slice.split(":")[1])
        train_structures = all_structures[train_index_begin:train_index_end]
        test_index_begin = int(test_slice.split(":")[0])
        test_index_end = int(test_slice.split(":")[1])
        test_structures = all_structures[test_index_begin:test_index_end]
        print("Shuffling and extraction done")

    return train_structures, test_structures