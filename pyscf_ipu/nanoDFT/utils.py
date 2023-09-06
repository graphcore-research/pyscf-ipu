# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import sys
import h5py
import pubchempy
import numpy as np
from itertools import combinations
from operator import itemgetter


spice_amino_acids = [
    "TRP", "LYN", "TYR", "PHE", "LEU", 
    "ILE", "HIE", "MET", "GLN", "HID", 
    "GLH", "VAL", "GLU", "THR", "PRO", 
    "ASN", "ASH", "ASP", "SER", "CYS", 
    "CYX", "ALA", "GLY"
]

def open_spice_amino_acids_hdf5():
    """Returns a h5py File object for the solvated amino acids data set in SPICE.
    Downloads the data set from github (1.4MB) if the file does not exist in the current directory."""
    import os.path

    spice_aa_fn = "solvated-amino-acids.hdf5"
    spice_aa_permalink = "https://github.com/openmm/spice-dataset/raw/e4e4ca731a8094b9a448d9831dd05de29124bfd9/solvated-amino-acids/solvated-amino-acids.hdf5"

    if not os.path.exists(spice_aa_fn):
        from urllib import request

        request.urlretrieve(spice_aa_permalink, spice_aa_fn)
    
    f_aa = h5py.File(spice_aa_fn)

    return f_aa
        

def get_mol_str_spice_aa(entry: str = "TRP", conformation: int = 0):
    """Returns the geometry for the amino acid in the 'entry' parameter.
    The data is extracted from the solvated amino acid data set in SPICE
    If the data set is not already available in the current dir, it is downloaded"""

    print(f"Getting geometry from the SPICE 'Solvated Amino Acids Dataset v1.1' for '{entry}'")
    f_aa = open_spice_amino_acids_hdf5()

    mol = f_aa[entry]
    nm_to_angstrom = 10.0
    return list(
        zip(
                [n for n in filter(str.isalpha, mol['smiles'][0].decode().upper())],
                mol['conformations'][conformation]*nm_to_angstrom
        )
    )

def get_mol_str_pubchem(entry: str):
    """Returns the geometry for the compound specified as 'entry' from the
    PubChem database.
    'entry' is interpreted as Compound ID if it is a string of digits or as
    name of a compound otherwise"""

    if entry.isdigit():                    # If all digits, we assume it is a CID
        print(f"Searching in PubChem for CID '{entry}'")
        compound = pubchempy.get_compounds(entry, "cid", record_type='3d')
    else:  
        print(f"Searching in PubChem for compound with name '{entry}'")                              # if not, we assume it is a name
        compound = pubchempy.get_compounds(entry, 'name', record_type='3d')
    mol_str = []
    if len(compound) > 1:
        print("INFO: PubChem returned more than one compound; using the first...", file=sys.stderr)
    elif len(compound) == 0:
        print(f"No compound found with the name '{entry}' in PubChem")
        return None
    print(f"Found compound: {compound[0].synonyms[0]}") 
    for a in compound[0].atoms:
        mol_str.append((a.element,np.array([a.x, a.y, a.z])))   
    return mol_str



def process_mol_str(mol_str: str):
    if mol_str == "benzene":
        mol_str = [
            ["C", ( 0.0000,  0.0000, 0.0000)],
            ["C", ( 1.4000,  0.0000, 0.0000)],
            ["C", ( 2.1000,  1.2124, 0.0000)],
            ["C", ( 1.4000,  2.4249, 0.0000)],
            ["C", ( 0.0000,  2.4249, 0.0000)],
            ["C", (-0.7000,  1.2124, 0.0000)],
            ["H", (-0.5500, -0.9526, 0.0000)],
            ["H", (-0.5500,  3.3775, 0.0000)],
            ["H", ( 1.9500, -0.9526, 0.0000)], 
            ["H", (-1.8000,  1.2124, 0.0000)],
            ["H", ( 3.2000,  1.2124, 0.0000)],
            ["H", ( 1.9500,  3.3775, 0.0000)]
        ]
    elif mol_str == "methane":
        mol_str = [
            ["C", (0, 0, 0)],
            ["H", (0, 0, 1)],
            ["H", (0, 1, 0)],
            ["H", (1, 0, 0)],
            ["H", (1, 1, 1)]
        ]
    elif mol_str in spice_amino_acids:
        mol_str = get_mol_str_spice_aa(mol_str)
    else:
        mol_str = get_mol_str_pubchem(mol_str)

    return mol_str


def min_interatomic_distance(mol_str):
    """This computes the minimum distance between atoms."""
    coords = map(itemgetter(1), mol_str) 
    distances = map(lambda x: np.linalg.norm(np.array(x[0]) - np.array(x[1])), combinations(coords, 2))
    return min(distances)