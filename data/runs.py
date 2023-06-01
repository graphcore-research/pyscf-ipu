import os 
from natsort import natsorted 
from tqdm import tqdm 
import pandas as pd 
folders = natsorted(os.listdir("../data/generated/run4/"))

dfs = []
mols = 0
pbar = tqdm(folders)
for folder in pbar:
  path =  "../data/generated/run4/%s/data.csv"%folder
  if os.path.isfile(path):  dfs.append(pd.read_csv(path)[1:]) # remove first one, it has compile time in it

df = pd.concat(dfs)

print(df.shape)
df.to_csv("gdb11_run4.csv")
