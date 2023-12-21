import wandb 
from rdkit import Chem
import rdkit 
import rdkit.Chem
import rdkit.Chem.AllChem
from rdkit.Geometry.rdGeometry import Point3D
from rdkit.Chem import AllChem
import numpy as np

def create_rdkit_mol(atom_types, atom_positions):
    mol = Chem.RWMol()
    for atom_type in atom_types:
        atom = Chem.Atom(atom_type)
        mol.AddAtom(atom)
    conf = Chem.Conformer(len(atom_types))
    for i, pos in enumerate(atom_positions):
        if isinstance(pos, np.ndarray): pos = pos.tolist()
        point = Point3D(*pos)
        conf.SetAtomPosition(i, point)
    mol.AddConformer(conf)
    return wandb.Molecule.from_rdkit(mol, convert_to_3d_and_optimize=False)
