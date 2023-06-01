#!source /nethome/alexm/poplar_sdk-ubuntu_20_04-3.1.0+1205-58b501c780/enable
#!bash ../source.sh #source /nethome/alexm/poplar_sdk-ubuntu_20_04-3.1.0+1205-58b501c780/enable
from jax.config import config
config.FLAGS.jax_platform_name = 'cpu'
import re 
from rdkit      import Chem
from rdkit.Chem import AllChem
from pyscf.gto.mole import Mole
import os
num_threads = 15
os.environ["OMP_NUM_THREADS"] = str(num_threads)
angstrom_to_bohr = 1.88973

def get_atom_string(atoms, locs):
    atom_string = atoms 
    atoms = re.findall('[a-zA-Z][^A-Z]*', atoms)
    str = ""
    for atom, loc in zip(atoms, locs): 
      str += "%s %4f %4f %4f; "%((atom,) + tuple(loc) )
    return atom_string, str 

smiles = [a for a in open("gdb/gdb11_size10_sorted.csv", "r").read().split("\n")]
smile = smiles[3000000//2:3000000//2+200][1]

print(">>>", smile)

atoms = [a for a in list(smile.upper()) if a == "C" or a == "N" or a == "O" or a == "F"]

b = Chem.MolFromSmiles(smile)
b = Chem.AddHs(b, explicitOnly=False)   

AllChem.EmbedMolecule(b)

atoms = [atom.GetSymbol() for atom in b.GetAtoms()]
num_hs = len([a for a in atoms if a == "H"])
print(num_hs)
locs =  b.GetConformer().GetPositions() * angstrom_to_bohr  

atom_string, string = get_atom_string(" ".join(atoms), locs)

mol = Mole() 
mol = mol.build(mol, atom=string, unit="Bohr", basis="sto3g", spin=0, verbose=0)


import os
os.environ["OMP_NUM_THREADS"] = "4"
import jax 
from jax import config 
config.update('jax_enable_x64', True) # perhaps it's the loss computation in the end? 
import time
import numpy 
import pyscf.dft
import numpy as np 
import numpy 
from pyscf import gto 
from pyscf import lib

GROUP_BOX_SIZE = 1.2
GROUP_BOUNDARY_PENALTY = 4.2
def arg_group_grids(mol, coords, box_size=GROUP_BOX_SIZE):
    '''
    Parition the entire space into small boxes according to the input box_size.
    Group the grids against these boxes.
    '''
    times = []
    times.append(time.time())
    atom_coords = mol.atom_coords()
    times.append(time.time())
    boundary = [atom_coords.min(axis=0) - GROUP_BOUNDARY_PENALTY, atom_coords.max(axis=0) + GROUP_BOUNDARY_PENALTY]
    times.append(time.time())
    # how many boxes inside the boundary
    boxes = ((boundary[1] - boundary[0]) * (1./box_size)).round().astype(int)
    times.append(time.time())
    #tot_boxes = numpy.prod(boxes + 2)
    #logger.debug(mol, 'tot_boxes %d, boxes in each direction %s', tot_boxes, boxes)
    # box_size is the length of each edge of the box
    box_size = (boundary[1] - boundary[0]) / boxes
    times.append(time.time())
    frac_coords = (coords - boundary[0]) * (1./box_size)
    times.append(time.time())
    box_ids = numpy.floor(frac_coords).astype(int)
    times.append(time.time())
    box_ids[box_ids<-1] = -1
    times.append(time.time())
    box_ids[box_ids[:,0] > boxes[0], 0] = boxes[0]
    box_ids[box_ids[:,1] > boxes[1], 1] = boxes[1]
    box_ids[box_ids[:,2] > boxes[2], 2] = boxes[2]
    times.append(time.time()) # this is the one that takes 20 ms? 
    rev_idx, counts = numpy.unique(box_ids, axis=0, return_inverse=True, return_counts=True)[1:3]
    times.append(time.time())
    times = np.array(times)
    #print(np.around(times[1:] - times[:-1], 2))
    return rev_idx.argsort(kind='stable')

from pyscf.dft import radi
import numpy 


from pyscf.data.elements import charge as elements_proton
def f(g): 
    g = (3 - g**2) * g * .5
    g = (3 - g**2) * g * .5
    g = (3 - g**2) * g * .5
    return g 

# vol is same size but coords increases a little in size. 
# could vmap this over natm if we pad coords! 
def iter(vol, ia, natm, ngrids, atm_dist, a, atm_coords, coords):
    import jax.numpy as np
    dcs                              = coords.reshape(1, -1, 3) - atm_coords.reshape(-1, 1, 3) + atm_coords[ia].reshape(1,1,3)
    grid_dist                        = np.linalg.norm(dcs, axis=2)
    pbecke                           = numpy.ones((natm,ngrids), dtype=np.float32)
    grid_dists                       = grid_dist.reshape(natm, 1, ngrids) - grid_dist.reshape(1, natm, ngrids)
    atm_grid_dists                   = (1/atm_dist).reshape(natm, natm, 1) * grid_dists
    atm_grid_dists_radii_adjusted    = (atm_grid_dists**2 - 1) * -a.reshape(natm, natm, 1) + atm_grid_dists
    atm_grid_dists_radii_adjusted    =  f(atm_grid_dists_radii_adjusted) 
    atm_grid_dists_radii_adjusted_m1 = 0.5*(1-atm_grid_dists_radii_adjusted)
    atm_grid_dists_radii_adjusted_p1 = 0.5*(1+atm_grid_dists_radii_adjusted)

    ones  =  numpy.ones((atm_grid_dists_radii_adjusted_p1.shape) )

    #atm_grid_dists_radii_adjusted_m1 = atm_grid_dists_radii_adjusted_m1.at[numpy.triu_indices(natm)].set(1)
    #atm_grid_dists_radii_adjusted_p1 = atm_grid_dists_radii_adjusted_p1.at[numpy.triu_indices(natm)].set(1)
    #atm_grid_dists_radii_adjusted_m1 = atm_grid_dists_radii_adjusted_m1 * zeros + ones 
    #atm_grid_dists_radii_adjusted_p1 = atm_grid_dists_radii_adjusted_p1 * zeros + ones 
    #>>> np.tril(a) + np.triu(np.ones((4,4)), k=1)
    atm_grid_dists_radii_adjusted_p1 = np.transpose(np.tril(np.transpose(atm_grid_dists_radii_adjusted_p1, (2,0,1)), k=-1) + np.triu( np.ones((natm,natm))) , (1,2,0))
    atm_grid_dists_radii_adjusted_m1 = np.transpose(np.tril(np.transpose(atm_grid_dists_radii_adjusted_m1, (2,0,1)), k=-1) + np.triu( np.ones((natm,natm))) , (1,2,0))

    pbecke *= np.prod(atm_grid_dists_radii_adjusted_m1, axis=1)
    pbecke *= np.prod(atm_grid_dists_radii_adjusted_p1, axis=0)

    norm = pbecke.sum(axis=0)
    weights = vol * pbecke[ia] * (1. / norm )
    return weights 


def iter2(vol, ia, natm, ngrids, atm_dist, a, atm_coords, coords):
    import jax.numpy as np
    dcs         = coords.reshape(1, -1, 3) - atm_coords.reshape(-1, 1, 3) + atm_coords[ia].reshape(1,1,3)
    grid_dist      = np.linalg.norm(dcs, axis=2)
    pbecke         = numpy.ones((natm,ngrids), dtype=np.float32)
    grid_dists     = grid_dist.reshape(natm, 1, ngrids) - grid_dist.reshape(1, natm, ngrids)
    atm_grid_dists                   = (1/atm_dist).reshape(natm, natm, 1) * grid_dists
    atm_grid_dists_radii_adjusted    = (atm_grid_dists**2 - 1) * -a.reshape(natm, natm, 1) + atm_grid_dists
    atm_grid_dists_radii_adjusted    =  f(atm_grid_dists_radii_adjusted) 
    atm_grid_dists_radii_adjusted_m1 = 0.5*(1-atm_grid_dists_radii_adjusted)
    atm_grid_dists_radii_adjusted_p1 = 0.5*(1+atm_grid_dists_radii_adjusted)

    #atm_grid_dists_radii_adjusted_m1 = atm_grid_dists_radii_adjusted_m1.at[numpy.triu_indices(natm)].set(1)
    #atm_grid_dists_radii_adjusted_p1 = atm_grid_dists_radii_adjusted_p1.at[numpy.triu_indices(natm)].set(1)

    atm_grid_dists_radii_adjusted_p1 = np.transpose(np.tril(np.transpose(atm_grid_dists_radii_adjusted_p1, (2,0,1)), k=-1) + np.triu( np.ones((natm,natm))) , (1,2,0))
    atm_grid_dists_radii_adjusted_m1 = np.transpose(np.tril(np.transpose(atm_grid_dists_radii_adjusted_m1, (2,0,1)), k=-1) + np.triu( np.ones((natm,natm))) , (1,2,0))


    pbecke *= np.prod(atm_grid_dists_radii_adjusted_m1, axis=1)
    pbecke *= np.prod(atm_grid_dists_radii_adjusted_p1, axis=0)

    norm = pbecke.sum(axis=0)
    weights = vol * pbecke[ia] * (1. / norm )
    return weights[0]

iter = jax.jit(iter, static_argnums=(2,3))
batched_iter = jax.vmap(iter, in_axes=(None, 0, None, None, None, None, None, 0), out_axes=(0))# for hyrodgen
iter2 = jax.jit(iter2, static_argnums=(2,3))
batched_iter_no_h = jax.vmap(iter2, in_axes=(0, 0, None, None, None, None, None, 0), out_axes=(0))


def joint(vol, h_indxs, natm, ngrids, atm_dist, a, atm_coords, h_coords_batch, noh_vols_batch, noh_indxs, noh_ngrids, noh_coords_batch):
    return batched_iter(vol, h_indxs, natm, ngrids, atm_dist, a, atm_coords, h_coords_batch), \
            batched_iter_no_h(noh_vols_batch, noh_indxs, natm, noh_ngrids, atm_dist, a, atm_coords, noh_coords_batch)

#joint = jax.jit(joint, static_argnums=(2,3,10), backend="ipu")
joint = jax.jit(joint, static_argnums=(2,3,10))

# TODO: refactor this to be jnp
# will be easy to rerwite, main problem will be figuring out how to have it nicely interact with jax.jit
def _get_partition(mol, atom_grids_tab,
                  atomic_radii=radi.BRAGG_RADII
                  ):
    ''' Generate the mesh grid coordinates and weights for DFT numerical integration.
    Returns:
        coord   (N, 3) 
        weights (N, ) 
    '''
    times = []
    times.append(time.time()) # 0 ms

    num_non_hs = len([a for a in mol.elements if a.lower() != "h"])
    charges = [elements_proton(x) for x in mol.elements]
    rad = numpy.sqrt(atomic_radii[charges]) + 1e-200
    rr = rad.reshape(-1,1) * (1./rad)
    a = .25 * (rr.T - rr)
    a[a<-.5] = -.5
    a[a>0.5] = 0.5

    times.append(time.time()) # 0 ms
    atm_coords  = mol.atom_coords() # (20, 3)
    atm_dist    = np.linalg.norm( atm_coords.reshape(-1, 1, 3) - atm_coords.reshape(1, -1, 3) , axis=2)
    coords_all  = []
    weights_all = []
    natm        = mol.natm
    times.append(time.time()) # 7ms 

    noh_vols_batch = []
    noh_coords_batch = []
    noh_sizes = []
    # reduced from 7->0ms by moving dcs computation into batched_iter.
    noh_ngrids = 1100 
    for ia in range(0, num_non_hs):  
        coords, vol = atom_grids_tab[mol.atom_symbol(ia)] 
        noh_sizes.append(coords.shape[0])
        noh_coords_batch.append( np.concatenate([coords, np.zeros((noh_ngrids-coords.shape[0], 3))], axis=0).reshape(1, -1))
        noh_vols_batch.append( np.concatenate((vol, np.zeros(noh_ngrids-vol.shape[0]))).reshape(1, -1) )

    noh_vols_batch = np.concatenate(noh_vols_batch)
    noh_indxs = np.arange(0, 10).reshape(-1, 1)
    noh_coords_batch = np.concatenate(noh_coords_batch)

    times.append(time.time())
    h_coords_batch = []
    for ia in range(num_non_hs, natm): 
        coords, vol = atom_grids_tab[mol.atom_symbol(ia)]  
        ngrids      = coords.shape[0]
        h_coords_batch.append(coords.reshape(1, -1))

    h_coords_batch = np.concatenate(h_coords_batch)
    h_indxs = np.arange(10, natm).reshape(-1, 1)
    times.append(time.time()) 

    inputs = [vol, h_indxs, natm, ngrids, atm_dist, a, atm_coords, h_coords_batch, noh_vols_batch, noh_indxs, noh_ngrids, noh_coords_batch]
    #inputs = [vol.astype(np.float32), h_indxs, natm, ngrids, atm_dist.astype(np.float32), a.astype(np.float32), atm_coords.astype(np.float32), 
    #          h_coords_batch.astype(np.float32), noh_vols_batch.astype(np.float32), noh_indxs, noh_ngrids, noh_coords_batch.astype(np.float32)]

    for a in inputs:
        try:
            print(a.dtype)
        except:
            print(a)

    _weights, w1 = joint(*inputs)

    #w1 = batched_iter_no_h(noh_vols_batch, noh_indxs, natm, noh_ngrids, atm_dist, a, atm_coords, noh_coords_batch)
    times.append(time.time())

    # 11 ms  
    for ia in range(0, num_non_hs):  weights_all.append( w1[ia].reshape(-1)[:noh_sizes[ia]] )

    # 1ms 
    for i, w in enumerate(_weights):
       weights_all.append(w.reshape(-1))

    for ia in range(0, natm):  # The first two here is just O(natm**2 * grid_size). 
        coords, vol = atom_grids_tab[mol.atom_symbol(ia)]  # there are only 5 different ones here (not 10)
        ngrids      = coords.shape[0]
        coords_all.append(coords+atm_coords[ia])


    times.append(time.time()) # 1ms 
    coords_all  = numpy.vstack(coords_all)
    weights_all = numpy.hstack(weights_all)
    #print(weights_all.shape)

    times.append(time.time())

    times = np.array(times)
    print("\t", np.around(times[1:] - times[:-1], 3), times[-1]-times[0])
    return coords_all, weights_all

def build_grid(self):
  mol = self.mol

  times = []
  times.append(time.time()) # below 0
  atom_grids_tab            = self.gen_atomic_grids( mol, self.atom_grid, self.radi_method, self.level, self.prune)
  times.append(time.time()) # below 0.13
  self.coords, self.weights = _get_partition(mol, atom_grids_tab, self.atomic_radii) 
  times.append(time.time()) # below is 0.02
  idx = arg_group_grids(mol, self.coords)
  times.append(time.time()) # below is 0

  self.coords  = self.coords[idx]
  self.weights = self.weights[idx]
  times.append(time.time()) # below is 0 

  # this actually does do smth?
  if self.alignment > 1:
      def _padding_size(ngrids, alignment):
          if alignment <= 1:
              return 0
          return (ngrids + alignment - 1) // alignment * alignment - ngrids

      padding = _padding_size(self.size, self.alignment)
      #logger.debug(self, 'Padding %d grids', padding)
      if padding > 0:
          self.coords = numpy.vstack(
              [self.coords, numpy.repeat([[1e4]*3], padding, axis=0)])
          self.weights = numpy.hstack([self.weights, numpy.zeros(padding)])
  
  times.append(time.time())
  self.screen_index = self.non0tab = None

  times = np.array(times)
  print(np.around(times[1:]-times[:-1], 3), times[-1]-times[0])

  return self

for _ in range(10**6):  
  t0 =time.time()
  grids1            = pyscf.dft.gen_grid.Grids(mol) 
  grids1.level      = 0 
  grids1.build()
  print(time.time()-t0)
  #print(grids1.coords.shape)

  t0 =time.time()
  grids2            = pyscf.dft.gen_grid.Grids(mol) 
  grids2.level      = 0 
  build_grid(grids2)
  #print(grids2.coords.shape)
  print(time.time()-t0)

  assert np.allclose(grids1.coords, grids2.coords)
  #print(np.max(np.abs(grids1.weights - grids2.weights)))
  assert np.allclose(grids1.weights, grids2.weights)