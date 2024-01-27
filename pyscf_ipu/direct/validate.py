import os
import time 
os.environ['OMP_NUM_THREADS'] = '29'
import jax
jax.config.update('jax_enable_x64', True)
import numpy as np 
import jax.numpy as jnp 
import scipy 
import pyscf 
from pyscf import gto, dft
import random 
random.seed(42)

def get_S(mol_str):
  m = pyscf.gto.Mole(atom=mol_str, basis="def2-svp", unit="bohr")
  m.build()
  return m.intor("int1e_ovlp")

if __name__ == "__main__":
  # most code below is copied form train.py; 
  # todo: refactor to re-use code. 
  class HashableNamespace:
    def __init__(self, namespace): self.__dict__.update(namespace.__dict__)
    def __hash__(self): return hash(tuple(sorted(self.__dict__.items())))

  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-basis',   type=str,   default="sto3g")  
  parser.add_argument('-level',   type=int,   default=0)
  parser.add_argument('-backend', type=str,       default="cpu") 
  parser.add_argument('-lr',      type=str,     default=5e-4) 
  parser.add_argument('-min_lr',      type=str,     default=1e-7)
  parser.add_argument('-warmup_iters',      type=float,     default=1000)
  parser.add_argument('-lr_decay',      type=float,     default=200000)
  parser.add_argument('-ema',      type=float,     default=0.0)
  parser.add_argument('-steps',   type=int,       default=100000)
  parser.add_argument('-bs',      type=int,       default=8)
  parser.add_argument('-val_bs',      type=int,   default=1)
  parser.add_argument('-mol_repeats',  type=int,  default=16) 
  parser.add_argument('-grad_acc', type=int, default=0) 
  parser.add_argument('-shuffle',  action="store_true") 
  parser.add_argument('-foriloop',  action="store_true") 
  parser.add_argument('-xc_f32',   action="store_true") 
  parser.add_argument('-eri_f32',  action="store_true") 
  parser.add_argument('-nn_f32',  action="store_true") 
  parser.add_argument('-eri_bs',  type=int, default=8) 
  parser.add_argument('-normal',     action="store_true") 
  parser.add_argument('-wandb',      action="store_true") 
  parser.add_argument('-prof',       action="store_true") 
  parser.add_argument('-visualize',  action="store_true") 
  parser.add_argument('-skip',       action="store_true", help="skip pyscf test case") 
  parser.add_argument('-nperturb',  type=int, default=0, help="how many atoms to perturb (supports 1,2,3)") 
  parser.add_argument('-qm9',        action="store_true") 
  parser.add_argument('-md17',       type=int, default=-1) 
  parser.add_argument('-qh9',        action="store_true") 
  parser.add_argument('-benzene',        action="store_true") 
  parser.add_argument('-hydrogens',        action="store_true") 
  parser.add_argument('-water',        action="store_true") 
  parser.add_argument('-waters',        action="store_true") 
  parser.add_argument('-alanine',        action="store_true") 
  parser.add_argument('-do_print',        action="store_true")  
  parser.add_argument('-states',         type=int,   default=1)
  parser.add_argument('-workers',        type=int,   default=5) 
  parser.add_argument('-precompute',        action="store_true")  
  parser.add_argument('-wiggle_var',     type=float,   default=0.05, help="wiggle N(0, wiggle_var), bondlength=1.5/30")
  parser.add_argument('-eri_threshold',  type=float,   default=1e-10, help="loss function threshold only")
  parser.add_argument('-rotate_deg',     type=float,   default=90, help="how many degrees to rotate")
  parser.add_argument('-test_dataloader',     action="store_true", help="no training, just test/loop through dataloader. ")
  parser.add_argument('-nn',       action="store_true", help="train nn, defaults to GD") 
  parser.add_argument('-tiny',     action="store_true") 
  parser.add_argument('-small',    action="store_true") 
  parser.add_argument('-base',     action="store_true") 
  parser.add_argument('-medium',   action="store_true") 
  parser.add_argument('-large',    action="store_true") 
  parser.add_argument('-xlarge',   action="store_true") 
  parser.add_argument('-largep',   action="store_true")  
  parser.add_argument('-inference',   default=0, type=int )  
  parser.add_argument('-forces',   action="store_true")  
  parser.add_argument("-checkpoint", default=-1, type=int, help="which iteration to save model (default -1 = no saving)") 
  parser.add_argument("-resume",   default="", help="path to checkpoint pickle file") 
  parser.add_argument("-name",   default="", help="substring of name to reload. ") 
  parser.add_argument('-md_T',  type=int,   default=300, help="temperature for md in Kelvin [K].")
  parser.add_argument('-md_time',  type=float,   default=0.002, help="time step for md in picoseconds [ps].")
  parser.add_argument('-train',  action="store_true") # evaluate on train set (validation set by default)
  parser.add_argument('-loss_vxc',  action="store_true") 
  opts = parser.parse_args()
  if opts.tiny or opts.small or opts.base or opts.large or opts.xlarge: opts.nn = True 
  import math 
  opts.lr = eval(opts.lr)
  opts.min_lr = eval(opts.min_lr)
  import sys 

  opts = HashableNamespace(parser.parse_args())
  sys.argv = sys.argv[:1]

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
  if opts.largep:  
      d_model= 91*16 
      n_heads = 16*1
      n_layers = 43
  if opts.xlarge:  
      d_model= 1600 
      n_heads = 25 
      n_layers = 48

  from train import nao 
  from transformer import transformer , transformer_init
  import pandas as pd 
  if opts.md17 == 1: df = pd.read_pickle("md17/val_water.pickle") 
  if opts.md17 == 2: df = pd.read_pickle("md17/val_ethanol.pickle") 
  if opts.md17 == 3: df = pd.read_pickle("md17/val_malondialdehyde.pickle") 
  if opts.md17 == 4: df = pd.read_pickle("md17/val_uracil.pickle") 
  print(df)
  import pickle
  print("searching for substring %s"%opts.name)
  from natsort import natsorted 
  folder = "checkpoints" # tmp folder, often delete everything when too large; automatically save them in saved_checkpoints
  #folder = "saved_checkpoints"
  all = os.listdir(folder)
  candidates = natsorted([a for a in all if opts.name in a])
  print("found candidates", candidates)
  to_load = candidates[-1].replace("_model.pickle", "").replace("_adam_state.pickle", "")
  print("loading candidate: ", candidates[-1])
  params = pickle.load(open("%s/%s_model.pickle"%(folder, to_load), "rb"))
  pickle.dump(params, open("saved_checkpoints/%s_model.pickle"%to_load, "wb"))

  rnd_key = jax.random.PRNGKey(42)
  n_vocab = nao("C", opts.basis) + nao("N", opts.basis) + \
            nao("O", opts.basis) + nao("F", opts.basis) + \
            nao("H", opts.basis) 

  rnd_key, cfg, _, total_params = transformer_init(
          rnd_key,
          n_vocab,
          d_model =d_model,
          n_layers=n_layers,
          n_heads =n_heads,
          d_ff    =d_model*4,
      )


  if opts.nn_f32: params = params.to_float32()
  else: params = params.to_float64()

  # create dataset like 
  # compute dummy output 
  from train import batched_state, summary, dm_energy
  valf = jax.jit(dm_energy, backend=opts.backend, static_argnames=("normal", 'nn', "cfg", "opts"))
  
  df = None 
  from torch.utils.data import DataLoader, Dataset
  class OnTheFlyQM9(Dataset):
      # prepares dft tensors with pyscf "on the fly". 
      # dataloader is very keen on throwing segfaults (e.g. using jnp in dataloader throws segfaul). 
      # problem: second epoch always gives segfault. 
      # hacky fix; make __len__ = real_length*num_epochs and __getitem__ do idx%real_num_examples 
      def __init__(self, opts, df=None, nao=294, train=True, num_epochs=10**9, extrapolate=False, do_pyscf=False):
          # df.sample is not deterministic; moved to pre-processing, so file is shuffled already. 
          # this shuffling is important, because it makes the last 10 samples iid (used for validation)
          #df = df.sample(frac=1).reset_index(drop=True) # is this deterministic? 
          if opts.qh9 or opts.qm9: 
              if train: self.mol_strs = df["pyscf"].values[:-10]
              else: self.mol_strs = df["pyscf"].values[-10:]
          #print(df["pyscf"].) # todo: print smile strings 
          
          self.num_epochs = num_epochs
          self.opts = opts 
          self.validation = not train 
          self.extrapolate = extrapolate
          self.do_pyscf = do_pyscf

          self.benzene = [
              ["C", ( 0.0000,  0.0000, 0.0000)],
              ["C", ( 1.4000,  0.0000, 0.0000)],
              ["C", ( 2.1000,  1.2124, 0.0000)],
              ["C", ( 1.4000,  2.4249, 0.0000)],
              ["C", ( 0.0000,  2.4249, 0.0000)],
              ["C", (-0.7000,  1.2124, 0.0000)],
              ["H", (-0.5500, -0.9526, 0.0000)],
              ["H", (-0.5500,  3.3775, 0.0000)],
              ["H", ( 1.9500, -0.9526, 0.0000)], 
              ["H", (-1.8000,  1.2124, 0.0000)],
              ["H", ( 3.2000,  1.2124, 0.0000)],
              ["H", ( 1.9500,  3.3775, 0.0000)]
          ]
          self.waters = [
              ["O",    (-1.464,  0.099,  0.300)],
              ["H",    (-1.956,  0.624, -0.340)],
              ["H",    (-1.797, -0.799,  0.206)],
              ["O",    ( 1.369,  0.146, -0.395)],
              ["H",    ( 1.894,  0.486,  0.335)],
              ["H",    ( 0.451,  0.165, -0.083)]
          ]

          if opts.benzene: self.mol_strs = [self.benzene]

          if opts.md17 > 0:  
            mol = {MD17_WATER: "water", MD17_ALDEHYDE: "malondialdehyde", MD17_ETHANOL: "ethanol", MD17_URACIL: "uracil"}[opts.md17]
            mode = {True: "train", False: "val"}[train]
            filename = "md17/%s_%s.pickle"%(mode, mol)
            print(filename)
            df = pd.read_pickle(filename)

            self.mol_strs = df["pyscf"].values.tolist()
            N = int(np.sqrt(df["H"].values.tolist()[0].reshape(-1).size))
            self.H = [a.reshape(N, N) for a in df["H"].values.tolist()]
            self.E = df["E"].values.tolist()
            self.mol_strs = [eval(a) for a in self.mol_strs]
            #self.do_pyscf = False # already pre-computed; still want valid ones to learn about augmentaiton. 
          else: 
              self.H = [np.zeros(1) for _ in self.mol_strs] # we have this for qh9; 
              self.E = [np.zeros(1) for _ in self.mol_strs]


          if opts.alanine: self.mol_strs = mol_str

          if train: self.bs = opts.bs 
          else: self.bs = opts.val_bs

              
      def __len__(self): return len(self.mol_strs)*self.num_epochs

      def __getitem__(self, idx):
          return batched_state(self.mol_strs[idx%len(self.mol_strs)], self.opts, self.bs, \
              wiggle_num=0, do_pyscf=self.do_pyscf, validation=False, \
                  extrapolate=self.extrapolate, mol_idx=idx), self.H[idx%len(self.mol_strs)], self.E[idx%len(self.mol_strs)]

  # difference with val_bs and bs
  mode = {True: "train", False: "valid"}[opts.train]
  val_qm9 = OnTheFlyQM9(opts, train=opts.train, df=df, do_pyscf=False)
  print(val_qm9)
   
  loss_E, loss_E2, loss_H, loss_eps = [], [], [], []
  E_us, E_pyscf, E_precomputed = [], [], []
  pyscf_judge = []
  pyscf_dff = []

  import pickle 
  from tqdm import tqdm 
  import matplotlib.pyplot as plt 
  fig, ax = plt.subplots()
  print(len(val_qm9))

  loader = DataLoader(val_qm9, batch_size=1, pin_memory=True, shuffle=False, drop_last=True, num_workers=5, prefetch_factor=2, collate_fn=lambda x: x[0])
  pbar = tqdm(enumerate(loader))
  t0 = time.time()
  old_H = None 
  for i, (state, H, E) in pbar: 
    t_load, t0 = time.time() - t0 , time.time()
    
    if i == 0: summary(state)
    state = jax.device_put(state)

    t_device, t0 = time.time() - t0 , time.time()

    _, (energies, losses, E_xc, density_matrix, vW, _H) = valf(params, state, opts.normal, opts.nn, cfg, opts)

    t_inference, t0 = time.time() - t0 , time.time()

    matrix = np.array(density_matrix[0])
    N = int(np.sqrt(matrix.size))

    E_precomputed.append(E*HARTREE_TO_EV)
    E_pyscf.append(np.array(state.pyscf_E).reshape(-1)[0])
    E_us.append(energies[0]*HARTREE_TO_EV)

    if opts.train: E = E[0][0]
    loss = np.abs(E*HARTREE_TO_EV- energies[0]*HARTREE_TO_EV)
    loss_E.append(loss)

    _loss = (E*HARTREE_TO_EV- energies[0]*HARTREE_TO_EV)
    if np.abs(E*HARTREE_TO_EV) < np.abs(energies[0]*HARTREE_TO_EV): 
        print("KASDJKSAJDLKASJDKLJASKJASKDJKALDJKLAJDKLJ", _loss)
        #exit()

    S = get_H_from_dm(matrix.reshape(N, N), val_qm9.mol_strs[i], state) 
    _val_H = _H[0]

    pred_vals  = scipy.linalg.eigh(_val_H, S)[0]
    label_vals = scipy.linalg.eigh(H, S)[0]
    eps = np.mean(np.abs(pred_vals - label_vals))
    H = np.mean(np.abs(H - _val_H)) 
    loss_eps.append(eps)
    loss_H.append(H) 

    loss_E2.append( np.abs(np.array(state.pyscf_E).reshape(-1)[0] - energies[0]*HARTREE_TO_EV) )

    pickle.dump([loss_eps, loss_H, loss_E2, loss_E, E_precomputed, E_us, E_pyscf], open("%s_evals.pickle"%mode, "wb"))


    pbar.set_description("E=%e E2=%e E_pscf=%e E_err=%e H=%e eps=%e ; tload=%4fs tdput=%f tnn=%f"%(np.mean(loss_E), np.mean(loss_E2), np.mean(pyscf_judge), np.mean(pyscf_dff), np.mean(loss_H), np.mean(loss_eps),
                                                                                                   t_load, t_device, t_inference))

    t0 = time.time() 