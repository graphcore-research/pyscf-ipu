import pickle
import numpy as np
import matplotlib.pyplot as plt


ml_file = "heatmap_data_009.pkl"
pyscf_file = "heatmap_pyscf_009.pkl"
# Load data from the pickle file
with open(ml_file, 'rb') as file:
    data_list = pickle.load(file)

with open(pyscf_file, 'rb') as file:
    pyscf_list = pickle.load(file)

# Extract phi, psi, and values from the loaded data
phi_values, psi_values, heatmap_val = zip(*data_list)

# Extract phi, psi, and values from the loaded data
phi_values_p, psi_values_p, heatmap_pyscf = zip(*pyscf_list)

matrix_size = int(len(data_list) ** 0.5)

heatmap_val = np.array(heatmap_val).reshape(matrix_size, matrix_size)
heatmap_pyscf = np.array(heatmap_pyscf).reshape(matrix_size, matrix_size)

# valid_E = NN(molecule) \approx E  
# state.pyscf_E = DFT(molecule) = E 
# state.valid_l = | NN(molecule) - DFT(molecule) | 
# 
heatmap_pyscf = -heatmap_pyscf

phi_coordinates, psi_coordinates = np.meshgrid(np.linspace(min(phi_values), max(phi_values), matrix_size),
                                               np.linspace(min(psi_values), max(psi_values), matrix_size))

fig, ax = plt.subplots(2,3, figsize=(10, 8))
# im = ax[0,0].imshow( heatmap_val )
im = ax[0,0].imshow(heatmap_val, cmap='viridis', origin='lower', extent=[min(psi_values), max(psi_values), min(phi_values), max(phi_values)])

# ax[0,0].set_xlim(phi_values)
# ax[0,0].set_ylim(psi_values)
im2 = ax[0,1].imshow( heatmap_pyscf, cmap='viridis', origin='lower', extent=[min(psi_values), max(psi_values), min(phi_values), max(phi_values)])
diff = ax[0,2].imshow( np.abs(heatmap_val - heatmap_pyscf), cmap='viridis', origin='lower', extent=[min(psi_values), max(psi_values), min(phi_values), max(phi_values)])

log = ax[1,0].imshow( np.log(np.abs(heatmap_val )), cmap='viridis', origin='lower', extent=[min(psi_values), max(psi_values), min(phi_values), max(phi_values)])
log2 = ax[1,1].imshow( np.log(np.abs(heatmap_pyscf )), cmap='viridis', origin='lower', extent=[min(psi_values), max(psi_values), min(phi_values), max(phi_values)])
difflog = ax[1,2].imshow( np.log(np.abs((heatmap_val - heatmap_pyscf))), cmap='viridis', origin='lower', extent=[min(psi_values), max(psi_values), min(phi_values), max(phi_values)])

for i in range(3):
    for j in range(2):
        ax[j, i].set_xticks(np.arange(phi_values[0], phi_values[-1], 45))
        ax[j, i].set_yticks(np.arange(psi_values[0], psi_values[-1], 45))
        # ax[j, i].set_xlim([phi_values[0], phi_values[-1]])
        # ax[j, i].set_ylim([psi_values[0], psi_values[-1]])
        ax[j, i].set_xlabel("phi [deg]")
        ax[j, i].set_ylabel("psi [deg]")

# orient = 'vertical'
orient = 'horizontal'
cbar = fig.colorbar(im, ax=ax[0, 0], orientation=orient, fraction=0.05, pad=0.28)
cbar = fig.colorbar(im2, ax=ax[0, 1], orientation=orient, fraction=0.05, pad=0.28)
cbar = fig.colorbar(diff, ax=ax[0, 2], orientation=orient, fraction=0.05, pad=0.28)
cbar = fig.colorbar(log, ax=ax[1, 0], orientation=orient, fraction=0.05, pad=0.28)
cbar = fig.colorbar(log2, ax=ax[1, 1], orientation=orient, fraction=0.05, pad=0.28)
cbar = fig.colorbar(difflog, ax=ax[1, 2], orientation=orient, fraction=0.05, pad=0.28)

# for a in ax.reshape(-1): a.axis("off")
ax[0,0].set_title("NN Energy")
ax[0,1].set_title("PySCF Energy")
ax[0,2].set_title("|NN-PySCF| Energy")

ax[1,0].set_title("NN log(|Energy|)")
ax[1,1].set_title("PySCF log(|Energy|)")
ax[1,2].set_title("|NN-PySCF| log(|Energy|)")
# ax[0,0].set_ylabel("Energy") # may fail with axis("off")
# ax[1,0].set_ylabel("log(|Energy|)")  # may fail with axis("off")
plt.tight_layout()

# Save the plot to a PNG file
plt.savefig("poc.png")