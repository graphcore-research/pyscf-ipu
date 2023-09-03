import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot4D(x, N, name=''):
    norm = mpl.colors.Normalize(vmin=np.min(x), vmax=np.max(x))

    fig, axs = plt.subplots(N, N)
    for i in range(N):
        for j in range(N):
            axs[i, j].imshow(x[i,j], norm=norm)
            axs[i, j].set_ylabel(f'i={i}')
            axs[i, j].set_xlabel(f'j={j}')
    
    for ax in axs.flat:
        ax.label_outer()

    fig.suptitle(name+' 4D (N, N, N, N)')
    plt.show()

def plot2D(x, N, name='', show_values=False):
    norm = mpl.colors.Normalize(vmin=np.min(x), vmax=np.max(x))
    fig, ax = plt.subplots()
    plt.imshow(x, norm=norm)
    if show_values:
        ixs, iys = np.meshgrid(np.arange(0, N, 1), np.arange(0, N, 1))
        for iy, ix in zip(iys.flatten(), ixs.flatten()):
            ax.text(iy, ix, x[ix, iy], va='center', ha='center') # indices swapped to match image

    fig.suptitle(name)
    plt.show()
