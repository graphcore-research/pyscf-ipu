import pandas as pd 
import numpy as np 
from rdkit import Chem
from rdkit.Chem import AllChem

df = pd.read_parquet("99_v2.parquet")
from tqdm import tqdm 

vals = []
pbar = tqdm(df["smiles"].values)
for smiles in  pbar: 
  #smiles = "C1CON(C2=NN3CC[C@@H]3CN2)C1"
  # create the molecule object from the SMILES string

  # regenerate this and don't, try to embed and check it's not -1 before proceeding? 
  mol = Chem.MolFromSmiles(smiles)
  mol = Chem.AddHs(mol)
  embed_result = AllChem.EmbedMolecule(mol)
  vals.append(embed_result)


  pbar.set_description( "%i %i"%(np.sum(vals), len(vals)  ))
