import numpy as np
import ase
from ase import io
import rascaline

def get_dataset_slices(dataset_path, train_slice, test_slice):
    
    if "rmd17" in dataset_path: # or "ch4" in dataset_path: or methane??
        print("Reading dataset")
        train_structures = ase.io.read(dataset_path, index = "0:1000")
        test_structures = ase.io.read(dataset_path, index = "1000:2000")
        print("Shuffling and extracting from dataset")
        np.random.shuffle(train_structures)
        np.random.shuffle(test_structures)
        train_index_begin = int(train_slice.split(":")[0])
        train_index_end = int(train_slice.split(":")[1])
        train_structures = train_structures[train_index_begin:train_index_end]
        test_index_begin = int(test_slice.split(":")[0])
        test_index_end = int(test_slice.split(":")[1])
        test_structures = test_structures[test_index_begin:test_index_end]
        print("Shuffling and extraction done")

    elif "methane" in dataset_path:
        print("Reading dataset")
        all_structures = ase.io.read(dataset_path, index = ":10000")
        print("Shuffling and extracting from dataset")
        np.random.shuffle(all_structures)
        train_index_begin = int(train_slice.split(":")[0])
        train_index_end = int(train_slice.split(":")[1])
        train_structures = all_structures[train_index_begin:train_index_end]
        test_index_begin = int(test_slice.split(":")[0])
        test_index_end = int(test_slice.split(":")[1])
        test_structures = all_structures[test_index_begin:test_index_end]
        print("Shuffling and extraction done")

    else:  # QM7 and QM9 don't seem to be shuffled randomly 
        print("Reading dataset")
        all_structures = ase.io.read(dataset_path, index = ":")
        print("Shuffling and extracting from dataset")
        np.random.shuffle(all_structures)
        train_index_begin = int(train_slice.split(":")[0])
        train_index_end = int(train_slice.split(":")[1])
        train_structures = all_structures[train_index_begin:train_index_end]
        test_index_begin = int(test_slice.split(":")[0])
        test_index_end = int(test_slice.split(":")[1])
        test_structures = all_structures[test_index_begin:test_index_end]
        print("Shuffling and extraction done")

    return train_structures, test_structures


def get_minimum_distance(structures):

    sd_hypers = {
        "cutoff": 5.0,
        "max_neighbors": 100,
        "separate_neighbor_species": False
    }
    sd_calculator = rascaline.SortedDistances(**sd_hypers)
    sds = sd_calculator.compute(structures)
    min_distance = 10.0
    for key, block in sds:
        min_distance_block = np.min(np.array(block.values))
        min_distance = min(min_distance, min_distance_block)
    print(f"The minimum distance in the dataset is {min_distance}")

    return min_distance

