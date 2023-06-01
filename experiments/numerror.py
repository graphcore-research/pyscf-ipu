# TODO
# print condition number for each matrix 
# print eigenvalues of each matrix 
# print loss at each iteration (float32, float64)


import numpy as np 
# optimize first for -skipdiis 
# add all the other things from xc to output! 
# E_xc, vhro, ... 
import matplotlib.pyplot as plt 
import matplotlib
import os 
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

import seaborn as sns
sns.set_theme()
sns.set_style("white")

fig, ax = plt.subplots(1, 1, figsize=(8,10))
float_name = {True: "float32", False: "float64"}

# add automatic plot after running! 
vals = []
xticks = []
xticklabels = []
iter_label = []

for outer_num, i in enumerate([0, 1, 34]): # plot energy aswell 
#for outer_num, i in enumerate([0, 1, 2, 24]):

  files32 = sorted([a for a in os.listdir("numerror/") if "[" not in a and int(a.split("_")[0]) == i and ".jpg" not in a and "True" in a and ".py" not in a and "allvals" not in a and "cs" not in a and "energy" not in a and "hyb" not in a and "n_electrons" not in a]  )
  files64 = sorted([a for a in os.listdir("numerror/") if "[" not in a and int(a.split("_")[0]) == i and ".jpg" not in a and "False" in a and ".py" not in a and "allvals" not in a and "cs" not in a and "energy" not in a and "hyb" not in a and "n_electrons" not in a]  )

  def prepare(val): 
    val = np.abs(val[val == val])
    val[np.logical_and(val<1e-15, val!=0)] = 2e-15 # show the ones that go out of plot 
    val[val==0] = 1e-17 # remove zeros. 
    #val = val[val > 1e-20]
    #val = val[val < 1e20]
    return val 

  # use the same sorting in here. 
  for num, (file32, file64) in enumerate(zip(files32, files64)):
    val32 = np.load("numerror/"+file32)["v"]
    val64 = np.load("numerror/"+file64)["v"]
    shape = val32.shape
    val32, val64 = prepare(val32), prepare(val64)
    print(val32.shape, val64.shape)

    indxs = np.argsort(val64)
    val32 = val32[indxs]
    val64 = val64[indxs]

    num_max_dots = 500 

    if val32.size > num_max_dots: val32 = val32[::int(val32.size)//num_max_dots] # plot 200 numbers for each thing 
    if val64.size > num_max_dots: val64 = val64[::int(val64.size)//num_max_dots]

    xs = -(np.ones(val64.shape[0])*num+outer_num*(3+len(files32)))
    #ax.plot([xs[0], xs[1]], [1e-15, 1e15], 'C%i-'%(num%10), lw=10, alpha=0.2)
    #ax.plot(xs, val64, 'C%io'%(num%10), ms=8, alpha=0.4)
    #ax.plot(xs, val32, 'kx', ms=2)
    ax.plot([1e-15, 1e18], [xs[0], xs[1]], 'C%i-'%(num%10), lw=10, alpha=0.2)
    ax.plot(val64, xs, 'C%io'%(num%10), ms=8, alpha=0.4)
    ax.plot(val32, xs, 'kx', ms=2)

    if num == 0: 
      iter_label.append((i, xs[0]))

    xticks.append(xs[0])
    xticklabels.append(file32.replace("_True.npz", "").replace("%i_"%i, ""))

#ax.set_yticks(np.arange(-16, 16)[])

plt.plot(0, 0, "C0o", label="float64", ms=8, alpha=0.4)
plt.plot(0, 0, "kx", label="float32", ms=2)


#plt.plot([0, xticks[-1]], [10**(-10), 10**(-10)], 'C7--', alpha=0.6)
#plt.plot([0, xticks[-1]], [10**(10), 10**10], 'C7--', alpha=0.6)
#plt.plot([0, xticks[-1]], [10**(0), 10**0], 'C7-', alpha=1)

plt.plot(
  [10**(-10), 10**(-10)], 
  [0, xticks[-1]], 
  'C7--', alpha=0.6)
plt.plot(
  [10**(10), 10**10], 
  [0, xticks[-1]], 
  'C7--', alpha=0.6)
plt.plot(
  [10**(0), 10**0], 
  [0, xticks[-1]], 
  'C7-', alpha=1)



#ax.annotate('SDL', xy=(0.5, 0.90), xytext=(0.5, 1.00), xycoords='axes fraction', 
#            fontsize=12, ha='center', va='bottom',
#            bbox=dict(boxstyle='square', fc='white'),
#            arrowprops=dict(arrowstyle='-[, widthB=7.0, lengthB=1.5', lw=2.0))
#ax.text(3, 100, "Hello, world!", fontsize=12, ha="center", va="center")
for x, label in zip(xticks, xticklabels): 
  plt.text(1e10, x-0.25, label, horizontalalignment='left', size='small', color='black', weight='normal')

for num , x in iter_label:
  plt.text(1e10, x+.6, "[Iteration %i]"%(num+1), horizontalalignment='left', size='small', color='black', weight='bold')

#plt.yaxis("off")
plt.yticks([], [])

plt.xscale("log")
plt.xlim([10**(-15), 10**18])
plt.legend()
#plt.axis("off")
#plt.xticks([10**i for i in range(-15, 16, 3)], range(-15,16, 3))
#plt.yticks(xticks, xticklabels, x=1e-10)#, rotation = 90)
plt.tight_layout()
plt.savefig("iteration1.jpg")