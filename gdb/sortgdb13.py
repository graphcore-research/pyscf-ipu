from tqdm import tqdm 
import os 
import time 
from rdkit import Chem  
from rdkit.Chem import AllChem
from rdkit import RDLogger
from tqdm import tqdm 
from natsort import natsorted 
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
import pandas as pd 
import sys

# run this on 10 different threads at the same time? 
#with open("13.smi", "r") as file: 
#  smiles = [ file.readline() for _ in tqdm(range(1000000000)) ] # this has 1 Billion 

#files = natsorted(os.listdir("/a/scratch/alexm/research/splitgdb13/"))

#n = 1000000000 // 100 # read 10 million? 
i = int(float(sys.argv[1]))
#print("/a/scratch/alexm/research/splitgdb13/%s"%files[i])

t0 = time.time()
#smiles = pd.read_csv("/a/scratch/alexm/research/splitgdb13/%s"%files[i]).values
n = 10000000
smiles = pd.read_csv("/a/scratch/alexm/research/gdb/13.cno.smi", header=None, skiprows=i*n, nrows=n).values
print(smiles.shape)
print(time.time()-t0)

import time 
t0 = time.time()

smiles_9 = []

num_hs = []
for j, smile in enumerate(tqdm(smiles)):
  smile = str(smile[0])
  atoms = [a for a in list(smile.upper()) if a == "C" or a == "N" or a == "O" or a == "F"]
  if len(atoms) != 13: continue 
  smiles_9.append(smile)
  b = Chem.MolFromSmiles(smile)
  b = Chem.AddHs(b) 
  atoms = [atom.GetSymbol() for atom in b.GetAtoms()]
  num_hs.append( len([a for a in atoms if a.upper() == "H"]))

  if j % 100000 == 0: 
    import pickle 
    with open("numhs/13cno_hs_%i.pkl"%i, "wb") as f: 
      pickle.dump(num_hs, f)
