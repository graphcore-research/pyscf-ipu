import numpy as np 
import os 
from natsort import natsorted
import wandb 

#wandb.init(project="jaxdft-numerror")

# plot all of these lines in wandb! 

str = ["allvals", "cs", "energies", "V_xc", "density_matrix", "last_density_matrix", "_V", "_H", "mf_diis_H", "vj", "vk", "eigvals", "eigvects", "energy", "overlap", "electron_repulsion", "fixed_hamiltonian", "L_inv", "_n_electrons_half", "weights", "hyb", "ao", "nuclear_energy"]

import matplotlib.pyplot as plt 

fig, ax = plt.subplots(5, 5, figsize=(30, 30))

ax = ax.reshape(-1)
first_ax = ax[0]
ax = ax[1:]

print(len(str))

maxs = []
used_str = []
num = 0 
for s in str: 


  if s == "allvalls": continue 
  if s == "cs": continue 
  if s == "energies": continue 
  #if s == "last_density_matrix": continue 
  if s == "overlap": continue 
  if s == "electron_repulsion": continue 
  if s == "fixed_hamiltonian": continue 
  if s == "L_inv": continue 
  if s == "_n_electrons_half": continue 
  if s == "weights": continue   # 3.8e-6 largest one of the constants
  if s == "hyb": continue 
  if s == "ao": continue  # 9e-7
  if s == "nuclear_energy": continue  # 3.6e-6 ; limit to how good we can get if we compute energy in float32
  used_str.append(s)

  

  diffs = []
  rel_diffs = []
  #for i in range(50):
  #for i in range(10):
  for i in range(5):


    float32 = np.load("numerror/%i_%s_%s.npz"%(i, s, True))["v"]
    float64 = np.load("numerror/%i_%s_%s.npz"%(i, s, False))["v"]
    print(i, s, np.max(np.abs(float64-float32)))

    if s == "V_xc": 
      indxs = np.argsort(float64.reshape(-1))
      ax[-i].plot(np.abs(float64.reshape(-1)[indxs]), 'bo-', ms=5)
      ax[-i].plot(np.abs(float32.reshape(-1)[indxs]), 'gx', ms=2)
      ax[-i].plot(np.abs((float64-float32).reshape(-1)[indxs]), 'r^', ms=2)
      ax[-i].set_yscale("log")

    diffs.append(np.max(np.abs(float32-float64)))
    rel_diffs.append(np.max(np.abs(float32-float64) / np.abs(float64)))

  print(num, ax.shape)
  ax[num].plot(diffs, '-gx', label=s)  
  ax[num].plot(rel_diffs, '-rx', label=s + "relative")  
  ax[num].set_yscale("log")
  ax[num].legend()

  maxs.append(np.max(diffs))
  num+=1


first_ax.plot(maxs, label="maxs")
first_ax.set_xticks(range(len(maxs)), used_str)
first_ax.set_yscale("log")
plt.tight_layout()
plt.savefig("experiments/numerror.jpg")
  


