import ase
from ase.io import read, write

structures = ase.io.read("methane.extxyz", ":100000")
ase.io.write("methane_reduced.extxyz", structures)
