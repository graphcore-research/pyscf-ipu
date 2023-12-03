import pandas as pd 
import os 
from rdkit import Chem
import numpy as np 
import pyscf
from natsort import natsorted 
from tqdm import tqdm 

# we test loading by reproducing labels with pyscf. 
# (instead of checking e.g. np_to_sdf) 

def sdf_to_np(filename):
  s = open('%s'%filename, 'r').read()
  lines = s.split('V2000')[1].split('\n')[1:-1]
  lines = [[a for a in line.split(' ') if a != '' ][:4] for line in lines if len(line)>35]
  lines = [[line[3], (float(line[0]), float(line[1]), float(line[2]))] for line in lines ]
  atom_str = [line[0] for line in lines]
  atom_pos = np.concatenate([np.array(line[1:]).reshape(1, -1) for line in lines ] )
  return atom_str, atom_pos

def np_to_pyscf(str, xyz):
  atom_list = []

  for i, atom_type in enumerate(str):
      x, y, z = xyz[i]
      atom_list.append([atom_type, (x, y, z)])

  return atom_list

def spin(pyscf_format):
  try: 
    mol = pyscf.gto.Mole(atom=pyscf_format, basis="6-31g(2df,p)")
    mol.build()
    return 0 
  except:
    mol = pyscf.gto.Mole(atom=pyscf_format, basis="6-31g(2df,p)", spin=1)
    mol.build()
    return 1

def nao(pyscf_format, spin):
  mol = pyscf.gto.Mole(atom=pyscf_format, basis="6-31g(2df,p)", spin=spin)
  mol.build()
  return mol.nao_nr()

# load all labels in the final 200k version 
df = pd.read_csv("final_version.csv")

# add info on train/test/val split 
train = pd.read_csv("dev/dev_target.csv")
valid = pd.read_csv("valid/valid_target.csv")
test  = pd.DataFrame({"gdb_idx": os.listdir("test/sdf/atom_11") + os.listdir("test/sdf/atom_12")})
df["train"] = df["gdb_idx"].isin(train["gdb_idx"])
df["test"]  = df["gdb_idx"].isin(test["gdb_idx"])
df["valid"] = df["gdb_idx"].isin(valid["gdb_idx"])

# alchemy computes  u0 = results['E_0K' ]  = (E0 + ZPE, 'Eh'), so need to subtract zpve
# https://github.com/tencent-alchemy/alchemy-pyscf/blob/fa4f7ff46be308302ba1e95754701142b6c4bf7f/alchemypyscf/thermo.py#L215
df["energy"] = df["U0\n(Ha, internal energy at 0 K)"] - df["zpve\n(Ha, zero point vibrational energy)"]

for folder in ["atom_9", "atom_10", "atom_11", "atom_12"]:
  files = natsorted(os.listdir(folder))

  strs, xyzs, pyscfs, gdb_idxs, naos, spins = [], [], [], [], [], []

  for f in tqdm(files): 
    try: 
      str, xyz = sdf_to_np("%s/%s"%(folder, f))
      strs.append(str)
      xyzs.append(xyz)
      pyscfs.append( np_to_pyscf(str, xyz) )
      gdb_idxs.append(int(f.replace(".sdf", "")))
      spins.append(spin(pyscfs[-1]))
      naos.append(-1)#nao(pyscfs[-1], spins[-1]))
    except: 
      print("broke %s"%f)

  df2 = pd.DataFrame({"gdb_idx": gdb_idxs, "pyscf": pyscfs, "str": strs, "xyz": xyzs, "nao": naos, "spin": spins})
  merged = pd.merge(df, df2, on="gdb_idx", how="inner")
    
  merged.to_pickle("%s.pickle"%folder)
  break