import pandas as pd 
import re 
import numpy as np 
import pyscf 
from tqdm import tqdm 
molecules = pd.read_pickle("../data/unique.pkl")
from pyscf import scf
    
def get_atom_string(atoms, locs):
    atom_string = atoms 
    atoms = re.findall('[a-zA-Z][^A-Z]*', atoms)
    str = ""
    for atom, loc in zip(atoms, locs): 
      str += "%s %4f %4f %4f; "%((atom,) + tuple(loc) )
    return atom_string, str 

# So PCQ uses eV and PySCF uses Hartree 
hartree_to_eV    = 27.2114
angstrom_to_bohr = 1.88973

ids = molecules["sdf_id"].iloc

print(molecules.shape)
import time 

time_sto3g = []
lst_sto3g = [ ]
lst_631g = [ ]
lst_631gs = [ ]
lst_def2 = [ ]
xs_sto3g= []
xs_631g= []
xs_631gs= []
xs_def2= []
mol = pyscf.gto.mole.Mole()
mol.verbose = 0 
mol.build(atom="C 0 0 0; C 0 0 1;", unit="Bohr", basis="sto3g", spin=0, verbose=0) # all time goes into gc.collect! 
mol.max_cycle = 100 


for i in range(0, molecules.shape[0], 100):
  atoms = molecules["atom_string"][ids[i]]
  locs  = molecules["atom_locations"][ids[i]]*angstrom_to_bohr

  atom_string, _str = get_atom_string(atoms, locs)

  for basis in ["STO3G", "6-31G", "6-31G*", "def2-TZVPPD"]:
    mol.build(atom=_str, unit="Bohr", basis=basis, spin=0, verbose=0) 
    mf    = scf.RKS(mol) 
    mf.xc = "b3lyp"
    t0 =time.time()
    pyscf_energy    = mf.kernel()  
    t1 = time.time()
    time_sto3g.append(t1-t0)
    lst_sto3g.append(mol.nao_nr())
    xs_sto3g.append(i)
    print("%18s %20f [eV] %8f [s]  %12s  N=%i"%(atom_string, pyscf_energy*hartree_to_eV, t1-t0, basis, mol.nao_nr()))

 