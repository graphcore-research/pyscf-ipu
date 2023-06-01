import time 
import numpy as np 
import pandas as pd 
import pyscf 
from pyscf.gto.mole import Mole
import re 

def get_atom_string(atoms, locs):
    atom_string = atoms 
    atoms = re.findall('[a-zA-Z][^A-Z]*', atoms)
    str = ""
    for atom, loc in zip(atoms, locs): 
      str += "%s %4f %4f %4f; "%((atom,) + tuple(loc) )
    return atom_string, str 

gdb9  = ["C"*20] 

'''

# level 0 

lexm@alexbow:~/jaxdft-pre-experimental/experiments/timings$ python time_c20_pyscf.py
100
7.829683065414429 s -723.3138792546174 sto-3g 100
180
11.047500610351562 s -736.9934161665187 6-31G 180
280
99.32744145393372 s -737.5771785796785 6-31G* 280

# with numa_node script; looks like it's a little slower for the large one 

100
7.068816184997559 s -723.313879254621 sto-3g 100
180
9.812366724014282 s -736.9934161664123 6-31G 180
280
117.26126074790955 s -737.5771785794711 6-31G* 280


#level 3


(without numa node thingy) [17:33:20] Molecule does not have explicit Hs.
Consider calling AddHs()
100
24.502097129821777 s -723.3660688780483 sto-3g 100
[17:33:44] Molecule does not have explicit Hs. Consider calling AddHs()
180
33.78191614151001 s -736.927278080333 6-31G 180
[17:34:18] Molecule does not have explicit Hs. Consider calling AddHs()
280
128.42879152297974 s -737.5048150776954 6-31G* 280

'''


    
from tqdm import tqdm 
from pyscf import gto, scf
from pyscf import __config__ 
__config__.dft_rks_RKS_grids_level = 3
import os 

for smile in [gdb9[0]]: 
  for basis in ["sto-3g", "6-31G", "6-31G*"]:
    atoms = [a for a in list(smile) if a == "C" or a == "N" or a == "O" or a == "F"]
    from rdkit import Chem  
    from rdkit.Chem import AllChem
    b = Chem.MolFromSmiles(smile)
    #b = Chem.AddHs(b) 
    AllChem.EmbedMolecule(b)

    atoms = [atom.GetSymbol() for atom in b.GetAtoms()]
    locs =  b.GetConformer().GetPositions()

    atom_string, string = get_atom_string(" ".join(atoms), locs)

    try:
      _mol = Mole(atom=string, unit='Bohr', basis=basis,  spin=0) 
      _mol.verbose = 0 
      _mol.build()
    except:
      continue 

    print(_mol.nao_nr())
      
        
    _mol.max_cycle = 50
    mf = scf.RKS(_mol) 
    mf.xc = "b3lyp"
    mf.diis_space = 8 

    t0 = time.time()
    pyscf_energy    = mf.kernel()  
    pyscf_time = time.time()-t0
    print(pyscf_time, "s", pyscf_energy, basis, _mol.nao_nr())


'''

[16:19:52] Molecule does not have explicit Hs. Consider calling AddHs()
10.178521156311035 s -325.8981140058157 sto-3g 45
[16:20:02] Molecule does not have explicit Hs. Consider calling AddHs()
14.501192569732666 s -331.6632166797748 6-31G 81
[16:20:17] Molecule does not have explicit Hs. Consider calling AddHs()
16.437588214874268 s -332.0745229359578 6-31G* 126
[16:20:33] Molecule does not have explicit Hs. Consider calling AddHs()
12.498451948165894 s -361.7615686765914 sto-3g 50
[16:20:46] Molecule does not have explicit Hs. Consider calling AddHs()
15.663350820541382 s -367.9693026921675 6-31G 90
[16:21:02] Molecule does not have explicit Hs. Consider calling AddHs()
17.75219464302063 s -368.3509005927019 6-31G* 140
[16:21:19] Molecule does not have explicit Hs. Consider calling AddHs()
16.051669597625732 s -397.7969270619933 sto-3g 55
[16:21:36] Molecule does not have explicit Hs. Consider calling AddHs()
17.100021600723267 s -404.61138353949093 6-31G 99
[16:21:53] Molecule does not have explicit Hs. Consider calling AddHs()
21.983044147491455 s -404.9970869501707 6-31G* 154




'''
