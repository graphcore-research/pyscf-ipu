import numpy as np 
import numpy 
from pyscf import gto 

GROUP_BOX_SIZE = 1.2
GROUP_BOUNDARY_PENALTY = 4.2
def arg_group_grids(mol, coords, box_size=GROUP_BOX_SIZE):
    '''
    Parition the entire space into small boxes according to the input box_size.
    Group the grids against these boxes.
    '''
    import numpy 
    atom_coords = mol.atom_coords()
    boundary = [atom_coords.min(axis=0) - GROUP_BOUNDARY_PENALTY, atom_coords.max(axis=0) + GROUP_BOUNDARY_PENALTY]
    # how many boxes inside the boundary
    boxes = ((boundary[1] - boundary[0]) * (1./box_size)).round().astype(int)
    tot_boxes = numpy.prod(boxes + 2)
    #logger.debug(mol, 'tot_boxes %d, boxes in each direction %s', tot_boxes, boxes)
    # box_size is the length of each edge of the box
    box_size = (boundary[1] - boundary[0]) / boxes
    frac_coords = (coords - boundary[0]) * (1./box_size)
    box_ids = numpy.floor(frac_coords).astype(int)
    box_ids[box_ids<-1] = -1
    box_ids[box_ids[:,0] > boxes[0], 0] = boxes[0]
    box_ids[box_ids[:,1] > boxes[1], 1] = boxes[1]
    box_ids[box_ids[:,2] > boxes[2], 2] = boxes[2]
    rev_idx, counts = numpy.unique(box_ids, axis=0, return_inverse=True, return_counts=True)[1:3]
    return rev_idx.argsort(kind='stable')

from pyscf.dft import radi
import numpy 

def original_becke(g):
    '''Becke, JCP 88, 2547 (1988); DOI:10.1063/1.454033'''
    g = (3 - g**2) * g * .5
    g = (3 - g**2) * g * .5
    g = (3 - g**2) * g * .5
    return g


# TODO: refactor this to be jnp
# will be easy to rerwite, main problem will be figuring out how to have it nicely interact with jax.jit
def _get_partition(mol, atom_grids_tab,
                  radii_adjust=None, atomic_radii=radi.BRAGG_RADII,
                  becke_scheme=original_becke, concat=True):
    '''Generate the mesh grid coordinates and weights for DFT numerical integration.
    We can change radii_adjust, becke_scheme functions to generate different meshgrid.

    Kwargs:
        concat: bool
            Whether to concatenate grids and weights in return

    Returns:
        grid_coord and grid_weight arrays.  grid_coord array has shape (N,3);
        weight 1D array has N elements.
    '''
    if callable(radii_adjust) and atomic_radii is not None:
        f_radii_adjust = radii_adjust(mol, atomic_radii)
    else:
        f_radii_adjust = None
    atm_coords = numpy.asarray(mol.atom_coords() , order='C')
    atm_dist = gto.inter_distance(mol)

    from pyscf import lib

    def gen_grid_partition(coords):
        ngrids = coords.shape[0]
        #grid_dist = numpy.empty((mol.natm,ngrids))
        grid_dist = numpy.empty((mol.natm,ngrids))
        for ia in range(mol.natm):
            dc = coords - atm_coords[ia]
            grid_dist[ia] = numpy.sqrt(numpy.einsum('ij,ij->i',dc,dc))
        pbecke = numpy.ones((mol.natm,ngrids))
        for i in range(mol.natm):
            for j in range(i):
                g = 1/atm_dist[i,j] * (grid_dist[i]-grid_dist[j])
                if f_radii_adjust is not None:
                    g = f_radii_adjust(i, j, g)
                #g = becke_scheme(g)# gets passed the one which returns None 
                g = original_becke(g)
                #print(g)
                pbecke[i] *= .5 * (1-g)
                pbecke[j] *= .5 * (1+g)
        return pbecke

    coords_all = []
    weights_all = []
    for ia in range(mol.natm):
        coords, vol = atom_grids_tab[mol.atom_symbol(ia)]
        coords = coords + atm_coords[ia]
        pbecke = gen_grid_partition(coords)
        weights = vol * pbecke[ia] * (1./pbecke.sum(axis=0))
        coords_all.append(coords)
        weights_all.append(weights)

    if concat:
        coords_all = numpy.vstack(coords_all)
        weights_all = numpy.hstack(weights_all)
    return coords_all, weights_all

def get_partition(self, mol, atom_grids_tab=None,
                      radii_adjust=None, atomic_radii=radi.BRAGG_RADII,
                      becke_scheme=original_becke, concat=True):
        if atom_grids_tab is None:
            atom_grids_tab = self.gen_atomic_grids(mol)
        return _get_partition(mol, atom_grids_tab, radii_adjust, atomic_radii, becke_scheme, concat=concat)

def build_grid(self):
  with_non0tab=False
  sort_grids=True
  mol = self.mol

  atom_grids_tab            = self.gen_atomic_grids( mol, self.atom_grid, self.radi_method, self.level, self.prune)

  self.coords, self.weights = get_partition(self, mol, atom_grids_tab, self.radii_adjust, self.atomic_radii, self.becke_scheme) 

  idx = arg_group_grids(mol, self.coords)
  self.coords  = self.coords[idx]
  self.weights = self.weights[idx]

  '''if self.alignment > 1:
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
  '''

  self.screen_index = self.non0tab = None

  return self