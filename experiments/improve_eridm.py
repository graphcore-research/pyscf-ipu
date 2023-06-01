import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
import os 
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
from jax.config import config
config.FLAGS.jax_platform_name = 'cpu'
import jax.numpy as jnp 

import seaborn as sns
sns.set_theme()
sns.set_style("white")


fig, ax = plt.subplots(1, 1, figsize=(8,20))
float_name = {True: "float32", False: "float64"}

# add automatic plot after running! 
vals        = []
xticks      = []
xticklabels = []
iter_label  = []

# I guess this kind of depends on ERI aswell! 
# looks like we aught to get there vk error is larger than the one we'll get here. 

hyb = 0.2
#for outer_num, i in enumerate([0, 1, 15, 24]): #,  4, 8, 16, 24]): # plot energy aswell 
for outer_num, i in enumerate(range(35)): 
  print("--- %i ---"%outer_num)
  files32 = sorted([a for a in os.listdir("numerror/") if "[" not in a and int(a.split("_")[0]) == i and ".jpg" not in a and "True" in a and ".py" not in a and "allvals" not in a and "cs" not in a and "energy" not in a and "hyb" not in a and "n_electrons" not in a]  )
  #files64 = sorted([a for a in os.listdir("numerror/") if "[" not in a and int(a.split("_")[0]) == i and ".jpg" not in a and "False" in a and ".py" not in a and "allvals" not in a and "cs" not in a and "energy" not in a and "hyb" not in a and "n_electrons" not in a]  )
  files32 = ["numerror/"+a for a in files32 if "electron_repulsion" in a or "density_matrix" in a]
  files64  = [a for a in files32]
  #print(files32)
  
  # we could even compute this using ipu and cmopare to that error; but we did htis and it's smth like 1e-6 so that should be okay? 
  # what about tricks like A@(b-c)=A@b - A@c  for c choosen to minimize error? 
  dm64      = np.load(files32[0])["v"].astype(np.float64) 
  #if i == 0: dm64 = dm64 / 100 
  eri64     = np.load(files32[1])["v"].astype(np.float64)
  vj64      = np.einsum('ijkl,ji->kl', eri64, dm64)  #/ c**2
  vk64      = np.einsum('ijkl,jk->il', eri64, dm64)*hyb  #/ args.scale_eri 
  vj_m_vk64 = vj64 - vk64 * hyb/2 

  dm32      = dm64.astype(np.float32)
  eri32     = eri64.astype(np.float32)
  vj32      = np.einsum('ijkl,ji->kl', eri32, dm32)  #/ c**2
  vk32      = np.einsum('ijkl,jk->il', eri32, dm32)*hyb  #/ args.scale_eri 
  vj_m_vk32 = vj32 - vk32 * hyb/2 

  d     = dm64.shape[0]
  dm32  = dm32.reshape(d**2)
  eri32 = eri32.reshape(d**2, d**2)

  err_vj = np.max(np.abs(vj64 - vj32))
  err_vk = np.max(np.abs(vk64 - vk32))
  err_diff = np.max(np.abs(vj_m_vk64 - vj_m_vk32))

  print("%20i %20i %10f %10f %10f"%(-1, -1, err_vj, err_vk, err_diff))


  # todo: 
  # compute both vj, vk and V_xc and report error (improvement) on all of them. 
  # I guess measure improvement in error and the HPO search scaling parametrs. 
  best_error = 10*10
  for i in range(5, 15): # this basically works, just need to fix first iteration 
    c = 3**i

    m = np.abs(eri32).max()*100
    I = np.eye(d**2, dtype=np.float32)

    
    # so scale initial density matrix down by 100x? 

    #vj_I = ( (eri32-np.eye(eri32.shape[0], dtype=np.float32)*eri32.max()/2)*c @ dm32.reshape(d**2)/m*c).reshape(d,d) 
    #vj32 = vj_I + dm32.reshape(d, d)*eri32.max()/2*c*c/m
    #vj32 = vj32 / c / c * m #* eri32.max() 

    # 1
    #vj_I = ((eri32 ) @ dm32.reshape(d**2)).reshape(d,d) 
    #vj32   = vj_I 

    # 2
    #vj_I = ((eri32 + I) @ dm32.reshape(d**2)).reshape(d,d) 
    #vj32   = vj_I - dm32.reshape(d, d)    

    # 3
    #vj_I = ((eri32 -np.diag(np.diag(eri32))) @ dm32.reshape(d**2)).reshape(d,d) 
    #vj32   = vj_I + (np.diag(eri32)*dm32.reshape(-1)).reshape(d, d)    

    # 4
    #dm_max = dm32.max()

    vj_I = ((eri32*c + I*c) @ dm32.reshape(d**2)).reshape(d,d) 
    vj32   = vj_I - dm32.reshape(d, d) * c 
    vj32 = vj32/c

    _eri32 = eri32.reshape(d,d,d,d).transpose(1,2,0,3).reshape(d**2, d**2)

    #print(np.unique(np.diag(_eri32)), np.unique(np.diag(eri32)))
    print(np.sort(np.diag(_eri32)), np.sort(np.diag(eri32)))
    #exit()


    vk_I = ((_eri32  + I / hyb*2 ) @ dm32.reshape(d**2) ).reshape(d,d) 
    vk32  = vk_I * hyb - dm32.reshape(d, d) *2 

    vj_m_vk32 = vj32 - vk32 * hyb / 2 
    #vj_m_vk32 = vj32 - vk32 * hyb/2 
    #vj_m_vk32 = vj_I - vk_I*hyb/2

    def ma(x): return np.max(np.abs(x))

    err_vj = ma(vj64 - vj32)
    err_vk = ma(vk64 - vk32)
    err_diff = ma(vj_m_vk32 - vj_m_vk64)
    print("%20i %20i %10f %10f %10f"%(c, i, err_vj, err_vk, err_diff), dm32.max())
    if err_vj < best_error: best_error = err_vj


  ax.plot( [1e-12, 1e12], [-outer_num, -outer_num], "C%i-"%(outer_num%10), lw=9, alpha=0.2)
  ax.plot( np.sort(np.abs(dm64.reshape(-1))), -np.ones(np.prod(dm32.shape))*outer_num, "C%io"%(outer_num%10), ms=9, alpha=0.6)
  ax.plot( np.sort(np.abs(dm32.reshape(-1))), -np.ones(np.prod(dm32.shape))*outer_num, "kx", ms=3)
  ax.text( 1e5, -outer_num, "[Iter%i] err=%10f"%(outer_num+1, best_error), horizontalalignment='left', size='small', color='black')#, weight='bold')
  #plt.axis("off")



  #print("symmetric:", np.allclose(dm64, dm64.T))  

plt.yticks([],[])
#for num , x in iter_label:
#  plt.text(1e10, x+.6, "[Iteration %i]"%(num+1), horizontalalignment='left', size='small', color='black', weight='bold')

plt.plot([0, 0], [0, -outer_num], 'k-', lw=3)
plt.xlim([10**(-15), 10**18])
plt.legend()


plt.xscale("log")
plt.xlim([1e-12, 1e12])
plt.tight_layout()
plt.savefig("eridm.jpg")



