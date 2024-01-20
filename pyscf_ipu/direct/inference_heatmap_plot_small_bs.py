import pickle 
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

HARTREE_TO_EV, EPSILON_B3LYP, HYB_B3LYP = 27.2114079527, 1e-20, 0.2

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-basis',   type=str,   default="sto3g")  
parser.add_argument('-level',   type=int,   default=0)

# GD options 
parser.add_argument('-backend', type=str,       default="cpu") 
parser.add_argument('-lr',      type=float,     default=2.5e-4)
parser.add_argument('-steps',   type=int,       default=100000)
parser.add_argument('-bs',      type=int,       default=8)
parser.add_argument('-val_bs',      type=int,   default=8)
parser.add_argument('-mol_repeats',  type=int,  default=16) # How many time to optimize wrt each molecule. 

# energy computation speedups 
parser.add_argument('-foriloop',  action="store_true") # whether to use jax.lax.foriloop for sparse_symmetric_eri (faster compile time but slower training. )
parser.add_argument('-xc_f32',   action="store_true") 
parser.add_argument('-eri_f32',  action="store_true") 
parser.add_argument('-eri_bs',  type=int, default=8) 

parser.add_argument('-normal',     action="store_true") 
parser.add_argument('-wandb',      action="store_true") 
parser.add_argument('-prof',       action="store_true") 
parser.add_argument('-visualize',  action="store_true") 
parser.add_argument('-skip',       action="store_true", help="skip pyscf test case") 

# dataset 
parser.add_argument('-qm9',        action="store_true") 
parser.add_argument('-benzene',        action="store_true") 
parser.add_argument('-hydrogens',        action="store_true") 
parser.add_argument('-water',        action="store_true") 
parser.add_argument('-waters',        action="store_true") 
parser.add_argument('-alanine',        action="store_true") 
parser.add_argument('-states',         type=int,   default=1)
parser.add_argument('-workers',        type=int,   default=5) 
parser.add_argument('-precompute',        action="store_true")  # precompute labels; only run once for data{set/augmentation}.
    # do noise schedule, start small slowly increase 
parser.add_argument('-wiggle_var',     type=float,   default=0.05, help="wiggle N(0, wiggle_var), bondlength=1.5/30")
parser.add_argument('-eri_threshold',  type=float,   default=1e-10, help="loss function threshold only")
parser.add_argument('-rotate_deg',     type=float,   default=90, help="how many degrees to rotate")

# models 
parser.add_argument('-nn',       action="store_true", help="train nn, defaults to GD") 
parser.add_argument('-tiny',     action="store_true") 
parser.add_argument('-small',    action="store_true") 
parser.add_argument('-base',     action="store_true") 
parser.add_argument('-medium',   action="store_true") 
parser.add_argument('-large',    action="store_true") 
parser.add_argument('-xlarge',   action="store_true") 

parser.add_argument("-checkpoint", default=-1, type=int, help="which iteration to save model (default -1 = no saving)") # checkpoint model 
parser.add_argument("-resume",   default="", help="path to checkpoint pickle file") # checkpoint model 

# inference heatmap plot args
parser.add_argument("-heatmap_step",   type=int,       default=10)
parser.add_argument("-plot_range",   type=int,       default=360)
opts = parser.parse_args()

# assert opts.val_bs * opts.heatmap_step == opts.plot_range, "[Temporary dependency] Try adjusting VAL_BS and HEATMAP_STEP so that their product is equal to PLOT_RANGE (by default 360)"
assert (opts.plot_range % (opts.val_bs * opts.heatmap_step)) == 0, "batch * step will not fit within the range with integer number of subranges"
if opts.tiny or opts.small or opts.base or opts.large or opts.xlarge: opts.nn = True 

if opts.alanine: 
    mol_str = [[ # 22 atoms (12 hydrogens) => 10 heavy atoms (i.e. larger than QM9). 
        ["H", ( 2.000 ,  1.000,  -0.000)],
        ["C", ( 2.000 ,  2.090,   0.000)],
        ["H", ( 1.486 ,  2.454,   0.890)],
        ["H", ( 1.486 ,  2.454,  -0.890)],
        ["C", ( 3.427 ,  2.641,  -0.000)],
        ["O", ( 4.391 ,  1.877,  -0.000)],
        ["N", ( 3.555 ,  3.970,  -0.000)],
        ["H", ( 2.733 ,  4.556,  -0.000)],
        ["C", ( 4.853 ,  4.614,  -0.000)], # carbon alpha 
        ["H", ( 5.408 ,  4.316,   0.890)], # hydrogne attached to carbon alpha 
        ["C", ( 5.661 ,  4.221,  -1.232)], # carbon beta 
        ["H", ( 5.123 ,  4.521,  -2.131)], # hydrogens attached to carbon beta 
        ["H", ( 6.630 ,  4.719,  -1.206)], # hydrogens attached to carbon beta 
        ["H", ( 5.809 ,  3.141,  -1.241)], # hydrogens attached to carbon beta 
        ["C", ( 4.713 ,  6.129,   0.000)],
        ["O", ( 3.601 ,  6.653,   0.000)],
        ["N", ( 5.846 ,  6.835,   0.000)],
        ["H", ( 6.737 ,  6.359,  -0.000)],
        ["C", ( 5.846 ,  8.284,   0.000)],
        ["H", ( 4.819 ,  8.648,   0.000)],
        ["H", ( 6.360 ,  8.648,   0.890)],
        ["H", ( 6.360 ,  8.648,  -0.890)],
    ]]

B, BxNxN, BxNxK = None, None, None
cfg = None
from train import dm_energy

from transformer import transformer_init
from train import nao
# global cfg
'''Model ViT model embedding #heads #layers #params training throughput
dimension resolution (im/sec)
DeiT-Ti N/A 192 3 12 5M 224 2536
DeiT-S N/A 384 6 12 22M 224 940
DeiT-B ViT-B 768 12 12 86M 224 292
Parameters Layers dmodel
117M 12 768
345M 24 1024
762M 36 1280
1542M 48 1600
'''
if opts.tiny:  # 5M 
    d_model= 192
    n_heads = 6
    n_layers = 12
if opts.small:
    d_model= 384
    n_heads = 6
    n_layers = 12
if opts.base: 
    d_model= 768
    n_heads = 12
    n_layers = 12
if opts.medium: 
    d_model= 1024
    n_heads = 16
    n_layers = 24
if opts.large: 
    d_model= 1280
    n_heads = 16
    n_layers = 36
if opts.xlarge: 
    d_model= 1600
    n_heads = 25
    n_layers = 48

if opts.nn: 
    rnd_key = jax.random.PRNGKey(42)
    n_vocab = nao("C", opts.basis) + nao("N", opts.basis) + \
              nao("O", opts.basis) + nao("F", opts.basis) + \
              nao("H", opts.basis)  
    rnd_key, cfg, params, total_params = transformer_init(
        rnd_key,
        n_vocab,
        d_model =d_model,
        n_layers=n_layers,
        n_heads =n_heads,
        d_ff    =d_model*4,
    )

# vandg = jax.jit(jax.value_and_grad(dm_energy, has_aux=True), backend=opts.backend, static_argnames=("normal", 'nn'))
valf = jax.jit(dm_energy, backend=opts.backend, static_argnames=("normal", 'nn', "cfg", "opts"))

from train import batched_state
from torch.utils.data import DataLoader, Dataset
class OnTheFlyQM9(Dataset):
    # prepares dft tensors with pyscf "on the fly". 
    # dataloader is very keen on throwing segfaults (e.g. using jnp in dataloader throws segfaul). 
    # problem: second epoch always gives segfault. 
    # hacky fix; make __len__ = real_length*num_epochs and __getitem__ do idx%real_num_examples 
    def __init__(self, opts, nao=294, train=True, num_epochs=10**9, extrapolate=False, init_phi_psi = None):
        # only take molecules with use {CNOFH}, nao=nao and spin=0.
        import pandas as pd 
        df = pd.read_pickle("alchemy/processed_atom_9.pickle") # spin=0 and only CNOFH molecules 
        if nao != -1: df = df[df["nao"]==nao] 
        # df.sample is not deterministic; moved to pre-processing, so file is shuffled already. 
        # this shuffling is important, because it makes the last 10 samples iid (used for validation)
        #df = df.sample(frac=1).reset_index(drop=True) # is this deterministic? 

        if train: self.mol_strs = df["pyscf"].values[:-10]
        else: self.mol_strs = df["pyscf"].values[-10:]
        #print(df["pyscf"].) # todo: print smile strings 
        
        self.num_epochs = num_epochs
        self.opts = opts 
        self.validation = not train 
        self.extrapolate = extrapolate
        self.init_phi_psi = init_phi_psi

        # self.benzene = [
        #     ["C", ( 0.0000,  0.0000, 0.0000)],
        #     ["C", ( 1.4000,  0.0000, 0.0000)],
        #     ["C", ( 2.1000,  1.2124, 0.0000)],
        #     ["C", ( 1.4000,  2.4249, 0.0000)],
        #     ["C", ( 0.0000,  2.4249, 0.0000)],
        #     ["C", (-0.7000,  1.2124, 0.0000)],
        #     ["H", (-0.5500, -0.9526, 0.0000)],
        #     ["H", (-0.5500,  3.3775, 0.0000)],
        #     ["H", ( 1.9500, -0.9526, 0.0000)], 
        #     ["H", (-1.8000,  1.2124, 0.0000)],
        #     ["H", ( 3.2000,  1.2124, 0.0000)],
        #     ["H", ( 1.9500,  3.3775, 0.0000)]
        # ]
        # self.waters = [
        #     ["O",    (-1.464,  0.099,  0.300)],
        #     ["H",    (-1.956,  0.624, -0.340)],
        #     ["H",    (-1.797, -0.799,  0.206)],
        #     ["O",    ( 1.369,  0.146, -0.395)],
        #     ["H",    ( 1.894,  0.486,  0.335)],
        #     ["H",    ( 0.451,  0.165, -0.083)]
        # ]

        # if opts.benzene: self.mol_strs = [self.benzene]
        # if opts.waters:  self.mol_strs = [self.waters]
        if opts.alanine: self.mol_strs = mol_str

        if train: self.bs = opts.bs 
        else: self.bs = opts.val_bs

    def __len__(self):
        return len(self.mol_strs)*self.num_epochs

    def __getitem__(self, idx):
        return batched_state(self.mol_strs[idx%len(self.mol_strs)], self.opts, self.bs, \
            wiggle_num=0, do_pyscf=self.validation or self.extrapolate, validation=False, \
                extrapolate=self.extrapolate, mol_idx=idx, init_phi_psi = self.init_phi_psi, inference=True, inference_psi_step=opts.heatmap_step)


print("loading checkpoint")
weights = pickle.load(open("%s_model.pickle"%opts.resume, "rb"))
print("done loading. ")

# print("loading adam state")
# adam_state = pickle.load(open("%s_adam_state.pickle"%opts.resume, "rb"))
# print("done")

# weights, adam_state = jax.device_put(weights), jax.device_put(adam_state)
weights = jax.device_put(weights)

from train import HashableNamespace

# make `opts` hashable so that JAX will not complain about the static parameter that is passed as arg
opts = HashableNamespace(opts)

data = []
pyscf = []
# data.append((1,1,344))
# data.append((2,4,323))
# data.append((3,3,334))
# data.append((4,2,331))

valid_E = None
val_state = None
for phi in range(0, opts.plot_range, opts.heatmap_step):
    # psi_start = 0
    # psi_end = psi_start + opts.val_bs * opts.heatmap_step
    # while psi_end <= opts.plot_range:
        # for psi in range(psi_start, psi_end, opts.heatmap_step):
    for psi in range(0, opts.plot_range, opts.val_bs * opts.heatmap_step):
        # print(psi, psi_start, psi_end, "<<<<<<<<<<<<<<<<<<")
        val_qm9 = OnTheFlyQM9(opts, train=False, init_phi_psi=(phi, psi))
        val_state = jax.device_put(val_qm9[0])
        # print("\n^^^^^^^^^^^\nJUST VAL QM9 [0]:", val_qm9[0])
        # print("WHOLE VAL QM9:", val_qm9)
        print("VAL_QM9[0].pyscf_E:", val_qm9[0].pyscf_E)
        _, (valid_vals, _, vdensity_matrix, vW) = valf(weights, val_state, opts.normal, opts.nn, cfg, opts)

        valid_l = np.abs(valid_vals*HARTREE_TO_EV-val_state.pyscf_E)
        valid_E = np.abs(valid_vals*HARTREE_TO_EV)

        print("valid_l: ", valid_l, "\nvalid_E: ", valid_E, "\nphi ", phi, " psi ", psi)

        for i in range(0, opts.val_bs):
            data.append((phi, psi + i * opts.heatmap_step, valid_E[i]))
            pyscf.append((phi, psi + i * opts.heatmap_step, val_state.pyscf_E[i].item()))
        # psi_start = 0 + psi_end
        # psi_end += opts.val_bs * opts.heatmap_step
            # data.append((phi, psi, valid_E[0]))

#data = np.log(np.abs(data))
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
# Extract phi, psi, and values from the data
phi_values, psi_values, heatmap_values = zip(*data)

# Define a grid
phi_grid, psi_grid = np.meshgrid(np.linspace(min(phi_values), max(phi_values), 100),
                                 np.linspace(min(psi_values), max(psi_values), 100))
# Interpolate values on the grid
heatmap_interpolated = griddata((phi_values, psi_values), heatmap_values, (phi_grid, psi_grid), method='cubic', fill_value=0)


# Create a filled contour plot
plt.contourf(psi_grid, phi_grid, heatmap_interpolated, cmap='viridis', levels=100)
plt.colorbar(label='Intensity')

# Set axis labels and title
plt.xlabel('Psi Angle')
plt.ylabel('Phi Angle')
plt.title('2D Heatmap with Interpolation')

# Save the plot to a PNG file
plt.savefig('heatmap_plot.png')

# Show the plot
plt.show()

import pickle

print("DATA ML", data)
print("DATA PYSCF", pyscf)
# Save data to a pickle file
with open('heatmap_data_bs2.pkl', 'wb') as file:
    pickle.dump(data, file)


# Save pyscf to a pickle file
with open('heatmap_pyscf_bs2.pkl', 'wb') as file:
    pickle.dump(pyscf, file)