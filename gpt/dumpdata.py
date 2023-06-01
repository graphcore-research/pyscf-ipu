import os 
from natsort import natsorted 
import pandas as pd 
import numpy as np 

# change this to save at /nethome/datasets/?

folders = natsorted([a for a in os.listdir("../data/generated/") if "." not in a ])[-16:]
dfs = []
from tqdm import tqdm
for folder in tqdm(folders):
  a = pd.read_csv("../data/generated/%s/data.csv"%folder, skiprows=2) 
  print(a.values.shape)
  print(a.head())
  print(a.shape)
  if np.prod(a.shape)>0:
    dfs.append(a.values)


#with open("gdb9_float64_cpu_3_23_2023.csv", "w", newline="") as csvfile: 
df = np.concatenate(dfs, axis=0) # pd.concat(dfs, axis=0, ignore_index=True, join="outer")
print(df.shape)
print(df[[5,10]])
#np.savez("gdb9_f32_ipu_3_23_2023.npz", data=df)
#np.savez("gdb9_f64_cpu_3_23_2023.npz", data=df)
np.savez("gdb9_f32_cpu_3_23_2023.npz", data=df)


