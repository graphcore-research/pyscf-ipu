# requires rdkit/pypi
import os
import os.path as osp
import rdkit
import numpy as np
import pandas as pd
from rdkit import Chem  # To extract information of the molecules
from rdkit.Chem import Draw  # To draw the molecules
import tarfile
import zipfile
from rdkit.Chem import AllChem

a = open("gdb11_size11.smi", "r").read().split("\n")
dct = {}

from tqdm import tqdm 
for count, b in enumerate(tqdm(a)):
  b = b.split("\t")[0].replace(" ", "").replace("(", ""). replace(")", "").replace("#", "").replace("=", "")
  b = [a for a in list(b) if a == "C" or a == "N" or a == "O" or a == "Cl" or a == "S"]
  a, c = np.unique(list(b), return_counts=True)

  key = str(a) + str(c)
  if key not in dct: 
    dct[key] = 1
  else: 
    dct[key] +=1
  
  if count % 100000 == 0: 
    print(dct)
  #print(a)
  #print(c)
  #exit()
  #print("[%s]"%b)
  '''b = Chem.MolFromSmiles(b)
  #b = Chem.AddHs(b) 
  AllChem.EmbedMolecule(b)
  print(b.GetConformer().GetPositions().shape)'''