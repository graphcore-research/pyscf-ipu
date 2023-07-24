# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger


lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

def sort_gdb(gdb_filename: str):
    smiles = [a.split("\t")[0] for a in open(gdb_filename, "r").read().split("\n")]
    # Keep onlysmiles with 9 heavy atoms
    atoms_count = 9
    smiles_filtered = []
    num_hs = []
    for smile in tqdm(smiles):
        atoms = [a for a in list(smile.upper()) if a == "C" or a == "N" or a == "O" or a == "F"]
        if len(atoms) != atoms_count: continue
        smiles_filtered.append(smile)
        b = Chem.MolFromSmiles(smile)
        b = Chem.AddHs(b)
        atoms = [atom.GetSymbol() for atom in b.GetAtoms()]
        num_hs.append( len([a for a in atoms if a.upper() == "H"]))

    # Sort by number of hydrogens.
    num_hs = np.array(num_hs)
    sorted_smiles = np.array(smiles_filtered)[np.argsort(num_hs)].tolist()
    df = pd.DataFrame(sorted_smiles[1:])
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Filter and sort GDB by number of atoms", epilog="Provide GDB .smi filename."
    )
    parser.add_argument("filename")
    args = parser.parse_args()

    gdb_filename = args.filename
    assert gdb_filename.endswith(".smi")
    gdb_sorted = sort_gdb(gdb_filename)
    # Save output as csv.
    out_filename = gdb_filename.replace(".smi", "_sorted.csv")
    gdb_sorted.to_csv(out_filename, index=False, header=False)

