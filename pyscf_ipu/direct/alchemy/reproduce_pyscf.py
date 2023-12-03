import pandas as pd 
import pyscf 
from pyscf import __config__ 
__config__.dft_rks_RKS_grids_level = 3
from pyscf import dft
import numpy as np 

df = pd.read_pickle("atom_9.pickle")
mol = pyscf.gto.Mole(atom=df["pyscf"].values[0], basis="6-31G(2df,p)", spin=0)
mol.build()
mf = pyscf.dft.RKS(mol)
mf.verbose = 4
mf.xc = 'B3LYP5' # pyscf changed b3lyp from vwn5 to vwn3 to agree with gaussian.
print(mf.kernel())
print(df["energy"].values[0])
print(df["homo"].values[0])
print(df["lumo"].values[0])
print(df["gap"].values[0])