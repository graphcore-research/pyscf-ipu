import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pyscf
import optax
from icecream import ic
from pyscf_ipu.exchange_correlation.b3lyp import b3lyp, vxc_b3lyp
from tqdm import tqdm 
import time 

HARTREE_TO_EV, EPSILON_B3LYP, HYB_B3LYP = 27.2114079527, 1e-20, 0.2

def T(x): return jnp.transpose(x, (0,2,1))

# Only need to recompute: L_inv, grid_AO, grid_weights, H_core, ERI and E_nuc. 
def dm_energy(W, state, diff_state, normal): 
    B, N, k        = W.shape
    L_inv_Q        = state.L_inv_T @ jnp.linalg.qr(W)[0]         # O(N^2 * num_electrons * batch) instead of O(N^3 * batch)! 
    density_matrix = 2 * L_inv_Q @ T(L_inv_Q) 
    E_xc           = exchange_correlation(density_matrix, state, diff_state, normal) 
    diff_JK        = JK(density_matrix, state, diff_state, normal)
    energies       = jnp.sum((density_matrix * (state.H_core + diff_JK/2)).reshape(B, -1), axis=-1) + E_xc + state.E_nuc
    return jnp.sum(energies), (energies, E_xc, density_matrix)

def exchange_correlation(density_matrix, state, diff_state, normal):
    B, _, gsize, N = state.grid_AO.shape
    if normal: 
        grid_AO_dm = (state.grid_AO[:, 0] @ density_matrix)         # (B,gsize,N) @ (N, N) = O(B gsize N^2)
        rho        = jnp.sum(grid_AO_dm * state.grid_AO , axis=3)   # (B,1,gsize,N) * (B,4,gsize,N) = O(B gsize N)
    else: 
        def sparse_mult(values, dm):
            in_ = dm.take(diff_state.cols, axis=0)
            prod = in_*values[:, None]
            return jax.ops.segment_sum(prod, diff_state.rows, gsize)

        main       = diff_state.main_grid_AO[:1, 0] @ density_matrix # (1, gsize, N) @ (N, N) = O(gsize N^2)
        correction = jax.vmap(sparse_mult)(diff_state.sparse_diffs_grid_AO, density_matrix)
        grid_AO_dm = (main - correction).reshape(B, 1, gsize, N)
        diff       = diff_state.main_grid_AO[:1, :] - diff_state.diffs_grid_AO
        rho        = jnp.sum(grid_AO_dm * diff, axis=3).reshape(B, 4, gsize)

    E_xc       = jax.vmap(b3lyp, in_axes=(0,None))(rho, EPSILON_B3LYP).reshape(B, gsize)
    E_xc       = jnp.sum(rho[:, 0] * state.grid_weights * E_xc, axis=-1).reshape(B)
    return E_xc 

def JK(density_matrix, state, diff_state, normal): 
    if normal: 
        J = jnp.einsum('bijkl,bji->bkl', state.ERI, density_matrix) 
        K = jnp.einsum('bijkl,bjk->bil', state.ERI, density_matrix) 
        diff_JK = J - K / 2 * HYB_B3LYP
    else: 
        from pyscf_ipu.nanoDFT.sparse_symmetric_ERI import sparse_symmetric_einsum
        # batched =>   flops = reads  
        #diff_JK = jax.vmap(sparse_symmetric_einsum, in_axes=(0, 0, 0))(state.nonzero_distinct_ERI, state.nonzero_indices, density_matrix)
        # first + correction_remaining =>  floats = reads*batch_size 
        diff_JK = jax.vmap(sparse_symmetric_einsum, in_axes=(None, None, 0))(state.nonzero_distinct_ERI[0], state.nonzero_indices[0], density_matrix)
        diff_JK = diff_JK - jax.vmap(sparse_symmetric_einsum, in_axes=(0, None, 0))(diff_state.diffs_ERI, diff_state.indxs, density_matrix)
 
    return diff_JK 


def nanoDFT(mol_str, opts, pyscf_E):
    # Init DFT tensors on CPU using PySCF.
    # Try to re-use grid amongst all points.
    state = init_dft(mol_str, opts)
    c, w = state.grid_coords, state.grid_weights
    print(mol_str[0][1])
    for _ in range(opts.bs-1):
        mol_str[0][1] = (mol_str[0][1][0]+0.05, mol_str[0][1][1], mol_str[0][1][2]) 
        stateB = init_dft(mol_str, opts, c, w)
        state = cat(state, stateB)
    N = state.N[0]

    summary(state)
    
    if opts.normal: diff_state = None 
    else: 
        main_grid_AO   = state.grid_AO[:1]
        diffs_grid_AO  = main_grid_AO - state.grid_AO
        rows, cols = np.nonzero(np.max(diffs_grid_AO[:, 0]!=0, axis=0))
        sparse_diffs_grid_AO = diffs_grid_AO[:, 0, rows,cols]

        diff_ERIs  = state.nonzero_distinct_ERI[:1] - state.nonzero_distinct_ERI
        diff_indxs = state.nonzero_indices[0].reshape(1, -1, 4)
        nzr        = np.abs(diff_ERIs[1]).reshape(-1) != 0
        diff_ERIs  = diff_ERIs[:, :, nzr]
        diff_indxs = diff_indxs[:, nzr]

        diff_state = DiffState(indxs=diff_indxs, 
        rows=rows, cols=cols,
                               main_grid_AO=main_grid_AO, sparse_diffs_grid_AO=sparse_diffs_grid_AO, diffs_grid_AO=diffs_grid_AO, diffs_ERI=diff_ERIs)
        summary(diff_state)

    if opts.visualize: 
        pass
    

    w = state.init 
    vandg = jax.jit(jax.value_and_grad( dm_energy, has_aux=True), backend=opts.backend, static_argnames=("normal", ))

    # Build initializers for params
    #adam = optax.adam(lr_schedule)
    adam = optax.adabelief(opts.lr)
    adam_state = adam.init(w)

    min_val = 0 
    min_dm  = 0 

    pbar = tqdm(range(opts.steps))

    (val, _), grad = vandg(w, state, diff_state, opts.normal)

    for i in pbar:
        #with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True): 
        (val, (vals, E_xc, density_matrix)), grad = vandg(w, state, diff_state, opts.normal)
        updates, adam_state = adam.update(grad, adam_state)
        w =  optax.apply_updates(w, updates)
        #pbar.set_description("energy=%.7f [eV] error=%.7f [eV] (best_error=%.7f[eV])"%(vals*HARTREE_TO_EV, target-vals[0]*HARTREE_TO_EV, target-min_val*HARTREE_TO_EV))

        if opts.bs == 1: pbar.set_description("error=%.7f [eV] (%.7f %.7f) "%(np.mean(val*HARTREE_TO_EV-state.pyscf_E), val*HARTREE_TO_EV, state.pyscf_E))
        else: 
            str = "error=" + "".join(["%.7f "%(vals[i]*HARTREE_TO_EV-state.pyscf_E[i]) for i in range(min(5,opts.bs))]) + " [eV]"
            #str += "E_xc=" + "".join(["%.7f "%(E_xc[i]*HARTREE_TO_EV) for i in range(opts.bs)]) + " [eV]"
            pbar.set_description(str)
        if i == 0: print("")
        
        if val < min_val: 
            min_val = val 
            min_dm = density_matrix

    val, density_matrix = min_val, min_dm

    # needs batching 
    exit()
    V_xc     = jax.grad(exchange_correlation)(density_matrix, state.grid_AO, state.grid_weights)
    V_xc     = (V_xc + V_xc.T)/2
    diff_JK  = get_JK(density_matrix, state.ERI)                
    H        = state.H_core + diff_JK + V_xc
    mo_energy, mo_coeff = np.linalg.eigh(state.L_inv @ H @ state.L_inv.T)
    mo_coeff = state.L_inv.T @ mo_coeff
    
    return val, (0, mo_energy, mo_coeff, state.grid_coords, state.grid_weights, density_matrix, H)


import chex
@chex.dataclass
class IterationState:
    init: np.array
    E_nuc: np.array
    mask: np.array
    L_inv: np.array
    L_inv_T: np.array
    H_core: np.array
    grid_AO: np.array
    grid_weights: np.array
    grid_coords: np.array
    pyscf_E: np.array
    N: int 
    ERI: np.array
    nonzero_distinct_ERI: np.array
    nonzero_indices: np.array

@chex.dataclass 
class DiffState: 
    diffs_ERI: np.array
    main_grid_AO: np.array
    diffs_grid_AO: np.array
    indxs: np.array
    sparse_diffs_grid_AO: np.array#jax.experimental.sparse.csr.CSR
    rows: np.array
    cols: np.array


from pyscf.data.elements import charge as elements_proton
from pyscf.dft import gen_grid, radi

def treutler_atomic_radii_adjust(mol, atomic_radii):
  charges = [elements_proton(x) for x in mol.elements]
  rad = np.sqrt(atomic_radii[charges]) + 1e-200
  rr = rad.reshape(-1, 1) * (1. / rad)
  a = .25 * (rr.T - rr)

  a[a < -0.5] = -0.5
  a[a > 0.5]  = 0.5
  a = jnp.array(a)

  def fadjust(i, j, g):
    g1 = g**2
    g1 -= 1.
    g1 *= -a[i, j]
    g1 += g
    return g1

  return fadjust


def inter_distance(coords):
  rr = np.linalg.norm(coords.reshape(-1, 1, 3) - coords, axis=2)
  rr[np.diag_indices(rr.shape[0])] = 0 
  return rr 

def original_becke(g):
  g = (3 - g**2) * g * .5
  g = (3 - g**2) * g * .5
  g = (3 - g**2) * g * .5
  return g

def get_partition(
  mol,
  atom_coords,
  atom_grids_tab,
  radii_adjust=treutler_atomic_radii_adjust,
  atomic_radii=radi.BRAGG_RADII,
  becke_scheme=original_becke,
  concat=True
):
  atm_dist = inter_distance(atom_coords)  # [natom, natom]

  def gen_grid_partition(coords):
    ngrids = coords.shape[0]
    dc = coords[None] - atom_coords[:, None]
    grid_dist = np.sqrt(np.einsum('ijk,ijk->ij', dc, dc))  # [natom, ngrid]

    ix, jx = np.tril_indices(mol.natm, k=-1)

    natm, ngrid = grid_dist.shape 
    g_ = -1 / atm_dist.reshape(natm, natm, 1) * (grid_dist.reshape(1, natm, ngrid) - grid_dist.reshape(natm, 1, ngrid))
    #g_ = jnp.array(g_)

    def pbecke_g(i, j):
      g = g_[i, j]
      charges = [elements_proton(x) for x in mol.elements]
      rad = np.sqrt(atomic_radii[charges]) + 1e-200
      rr = rad.reshape(-1, 1) * (1. / rad)
      a = .25 * (rr.T - rr)
      a[a < -0.5] = -0.5
      a[a > 0.5]  = 0.5
      g1 = g**2
      g1 -= 1.
      g1 *= -a[i, j].reshape(-1, 1)
      g1 += g
      return g1

    g = pbecke_g(ix, jx)
    g = np.copy(becke_scheme(g))
    gp2 = (1+g)/2
    gm2 = (1-g)/2

    pbecke = jnp.ones((mol.natm, ngrids))  # [natom, ngrid]
    pbecke = pbecke.at[ix].mul(gm2)
    pbecke = pbecke.at[jx].mul(gp2)

    return pbecke

  coords_all = []
  weights_all = []
  for ia in range(mol.natm):
    coords, vol = atom_grids_tab[mol.atom_symbol(ia)]
    coords = coords + atom_coords[ia]  # [ngrid, 3]
    pbecke = gen_grid_partition(coords)  # [natom, ngrid]
    weights = vol * pbecke[ia] / jnp.sum(pbecke, axis=0) 
    coords_all.append(coords)
    weights_all.append(weights)

  if concat:
    coords_all = jnp.vstack(coords_all)
    weights_all = jnp.hstack(weights_all)
  return coords_all, weights_all


class DifferentiableGrids(gen_grid.Grids):
  """Differentiable alternative to the original pyscf.gen_grid.Grids."""

  def build(self, atom_coords) :
    mol = self.mol

    atom_grids_tab = self.gen_atomic_grids(
      mol, self.atom_grid, self.radi_method, self.level, self.prune
    )

    coords, weights = get_partition(
      mol,
      atom_coords,
      atom_grids_tab,
      treutler_atomic_radii_adjust,
       self.atomic_radii,
      original_becke,
    )
    self.coords = coords
    self.weights = weights 
    return coords, weights


def grids_from_pyscf_mol(
  mol: pyscf.gto.mole.Mole, quad_level: int = 1
) :
  g = gen_grid.Grids(mol)
  g.level = quad_level
  g.build()
  grids = jnp.array(g.coords)
  weights = jnp.array(g.weights)
  return grids, weights


def init_dft(mol_str, opts, _coords=None, _weights=None):
    mol = build_mol(mol_str, opts.basis)
    pyscf_E, pyscf_hlgap, pycsf_forces = reference(mol_str, opts)

    N                = mol.nao_nr()                                 # N=66 for C6H6 (number of atomic **and** molecular orbitals)
    n_electrons_half = mol.nelectron//2                             # 21 for C6H6
    E_nuc            = mol.energy_nuc()                             # float = 202.4065 [Hartree] for C6H6. TODO(): Port to jax.

    from pyscf import dft
    #grids            = pyscf.dft.gen_grid.Grids(mol)
    grids            = DifferentiableGrids(mol)
    grids.level      = opts.level
    #grids.build()
    grids.build(np.concatenate([np.array(a[1]).reshape(1, 3) for a in mol._atom]))

    grid_weights    = grids.weights                                 # (grid_size,) = (45624,) for C6H6
    grid_coords     = grids.coords
    coord_str       = 'GTOval_cart_deriv1' if mol.cart else 'GTOval_sph_deriv1'
    grid_AO         = mol.eval_gto(coord_str, grids.coords, 4)      # (4, grid_size, N) = (4, 45624, 9) for C6H6.

    # TODO(): Add integral math formulas for kinetic/nuclear/O/ERI.
    kinetic         = mol.intor_symmetric('int1e_kin')              # (N,N)
    nuclear         = mol.intor_symmetric('int1e_nuc')              # (N,N)
    O               = mol.intor_symmetric('int1e_ovlp')             # (N,N)
    L               = np.linalg.cholesky(O)
    L_inv           = np.linalg.inv(L)          # (N,N)


    init = np.eye(N)[:, :n_electrons_half]
    #I_nxk = init[:, :n_electrons_half]

    mask = np.concatenate([np.ones(n_electrons_half), np.zeros(N-n_electrons_half)])
    if opts.normal: 
        ERI = mol.intor("int2e_sph")
        nonzero_distinct_ERI = np.zeros(1)
        nonzero_indices = np.zeros(1)
    else: 
        from pyscf_ipu.nanoDFT.sparse_symmetric_ERI import get_i_j, num_repetitions_fast
        eri_threshold = 0
        batches       = 1
        nipu          = 1
        distinct_ERI  = mol.intor("int2e_sph", aosym="s8")
        #below_thr = np.abs(distinct_ERI) <= eri_threshold
        #distinct_ERI[below_thr] = 0.0
        #ic(distinct_ERI.size, np.sum(below_thr), np.sum(below_thr)/distinct_ERI.size)
        #nonzero_indices      = np.nonzero(distinct_ERI)[0].astype(np.uint64)
        nonzero_indices      = np.arange(distinct_ERI.size)# ]np.nonzero(distinct_ERI)[0].astype(np.uint64)
        nonzero_distinct_ERI = distinct_ERI[nonzero_indices]#.astype(np.float32)

        ij, kl               = get_i_j(nonzero_indices)
        rep                  = num_repetitions_fast(ij, kl)
        nonzero_distinct_ERI = nonzero_distinct_ERI / rep
        remainder = nonzero_indices.shape[0] % (nipu*batches)

        if remainder != 0:
            ij = np.pad(ij, ((0,nipu*batches-remainder)))
            kl = np.pad(kl, ((0,nipu*batches-remainder)))
            nonzero_distinct_ERI = np.pad(nonzero_distinct_ERI, (0,nipu*batches-remainder))

        ij = ij.reshape(batches, -1)
        kl = kl.reshape(batches, -1)
        nonzero_distinct_ERI = nonzero_distinct_ERI.reshape(batches, -1)

        i, j = get_i_j(ij.reshape(-1))
        k, l = get_i_j(kl.reshape(-1))
        nonzero_indices = np.vstack([i,j,k,l]).T.reshape(batches, -1, 4)

        #ERI = [nonzero_distinct_ERI, nonzero_indices]
        #ERI = ERI 
        ERI = np.zeros(1)
        #ERI = mol.intor("int2e_sph")
        
    def e(x): return np.expand_dims(x, axis=0)

    
    state = IterationState(init = e(init), 
                           E_nuc=e(E_nuc), 
                           ERI=e(ERI),  
                           nonzero_distinct_ERI=e(nonzero_distinct_ERI),
                           nonzero_indices=e(nonzero_indices),
                           mask=e(mask), 
                           H_core=e(nuclear+kinetic),
                           L_inv=e(L_inv), 
                           L_inv_T = e(L_inv.T),
                           grid_AO=e(grid_AO), 
                           grid_weights=e(grid_weights), 
                           grid_coords=e(grid_coords),
                           pyscf_E=e(pyscf_E[-1:]), 
                           N=e(mol.nao_nr()),
                           )


    return state


def summary(state): 
    if state is None: return 
    print("_"*100)
    for field_name, field_def in state.__dataclass_fields__.items():
        field_value = getattr(state, field_name)
        try: 
            print("%20s %20s %20s"%(field_name,getattr(field_value, 'shape', None), getattr(field_value, "nbytes", None)/10**9))
        except: 
            print("BROKE FOR ", field_name)
    try:
        print(state.pyscf_E[:, -1])
    except:
        pass 
    print("_"*100)

def cat(dc1, dc2, axis=0):
    # Use dictionary comprehension to iterate over the dataclass fields
    concatenated_fields = {
        field: jnp.concatenate([getattr(dc1, field), getattr(dc2, field)], axis=axis)
        for field in dc1.__annotations__
    }
    # Create a new dataclass instance with the concatenated fields
    return IterationState(**concatenated_fields)


def grad_elec(weight, grid_AO, eri, s1, h1aos, natm, aoslices, mask, mo_energy, mo_coeff, mol, dm, H):
    # Electronic part of RHF/RKS gradients
    dm0  = 2 * (mo_coeff*mask) @ mo_coeff.T                                 # (N, N) = (66, 66) for C6H6.
    dme0 = 2 * (mo_coeff * mask*mo_energy) @  mo_coeff.T                    # (N, N) = (66, 66) for C6H6. 

    # Code identical to exchange correlation.
    rho             = jnp.sum( grid_AO[:1] @ dm0 * grid_AO, axis=2)         # (10, grid_size) = (10, 45624) for C6H6.
    _, vrho, vgamma = vxc_b3lyp(rho, EPSILON_B3LYP)                             # (grid_size,) (grid_size,)
    V_xc            = jnp.concatenate([vrho.reshape(1, -1)/2, 4*vgamma.reshape(1, -1)*rho[1:4]], axis=0)  # (4, grid_size)

    vmat = grid_AO[1:4].transpose(0, 2, 1) @ jnp.sum(grid_AO[:4] * jnp.expand_dims(weight * V_xc, axis=2), axis=0) # (3, N, N)
    aos = jnp.concatenate([jnp.expand_dims(grid_AO[np.array([1,4,5,6])], 0), jnp.expand_dims(grid_AO[np.array([2,5,7,8])], 0), jnp.expand_dims(grid_AO[np.array([3,6,8,9])], 0)], axis=0) # (3, N, N)
    V_xc = - vmat - jnp.transpose(jnp.einsum("snpi,np->spi", aos, weight*V_xc), axes=(0,2,1)) @ grid_AO[0]  # (3, 4, grid_size, N)

    vj = - jnp.einsum('sijkl,lk->sij', eri, dm0) # (3, N, N)
    vk = - jnp.einsum('sijkl,jk->sil', eri, dm0) # (3, N, N)
    vhf = V_xc + vj - vk * .5 * HYB_B3LYP        # (3, N, N)

    de = jnp.einsum('lxij,ij->lx', h1aos, dm0)   # (natm, 3)
    for k, ia in enumerate(range(natm)):
        p0, p1 = aoslices[ia][2], aoslices[ia][3]
        de = de.at[k].add(jnp.einsum('xij,ij->x', vhf[:, p0:p1], dm0[p0:p1]) * 2)
        de = de.at[k].add(-jnp.einsum('xij,ij->x', s1[:, p0:p1], dme0[p0:p1]) * 2)
    return de

def grad_nuc(charges, coords):
    # Derivatives of nuclear repulsion energy wrt nuclear coordinates
    natm = charges.shape[0]
    pairwise_charges    = charges.reshape(natm, 1) * charges.reshape(1, natm)                # (natm, natm)
    pairwise_difference = coords.reshape(1, natm, 3) - coords.reshape(natm, 1, 3)            # (natm, natm, 3)
    pairwise_distances  = jnp.linalg.norm(pairwise_difference, axis=2) ** 3                  # (natm, natm)
    pairwise_distances  = jnp.where(pairwise_distances == 0, jnp.inf, pairwise_distances)    # (natm, natm)
    all = - pairwise_charges.reshape(natm, natm, 1) * pairwise_difference                    # (natm, natm, 3)
    all = all / pairwise_distances.reshape(natm, natm, 1)                                    # (natm, natm, 3)
    all = all.at[jnp.diag_indices(natm)].set(0)                                              # (natm, natm, 3)
    return jnp.sum(all, axis=0)                                                              # (natm, natm)

def grad(mol, coords, weight, mo_coeff, mo_energy, dm, H):
    # Initialize DFT tensors on CPU using PySCF.
    ao = pyscf.dft.numint.NumInt().eval_ao(mol, coords, deriv=2)
    eri = mol.intor("int2e_ip1")
    s1  = - mol.intor('int1e_ipovlp', comp=3)
    kin = - mol.intor('int1e_ipkin',  comp=3)
    nuc = - mol.intor('int1e_ipnuc',  comp=3)

    mask = np.ones(mol.nao_nr())
    mask[mol.nelectron//2:] = 0

    aoslices = mol.aoslice_by_atom()
    h1 = kin + nuc
    def hcore_deriv(atm_id, aoslices, h1): # <\nabla|1/r|>
        _, _, p0, p1 = aoslices[atm_id]
        with mol.with_rinv_at_nucleus(atm_id):
            vrinv = mol.intor('int1e_iprinv', comp=3) #
            vrinv *= -mol.atom_charge(atm_id)
        vrinv[:,p0:p1] += h1[:,p0:p1]
        return vrinv + vrinv.transpose(0,2,1)
    N = h1.shape[1] # (3, N , N)
    h1aos = np.zeros((mol.natm, 3, N, N))
    for k, ia in enumerate(range(mol.natm)):
        p0, p1 = aoslices[ia,2:]
        h1aos[k] = hcore_deriv(ia, aoslices, h1)

    charges = np.zeros((mol.natm))
    coords = np.zeros((mol.natm,3))
    for j in range(mol.natm):
        charges[j] = mol.atom_charge(j)
        coords[j]= mol.atom_coord(j)

    #_grad_elec = jax.jit(grad_elec, static_argnames=["aoslices", "natm"], backend="cpu")
    _grad_elec = grad_elec
    _grad_nuc = jax.jit(grad_nuc, backend="cpu")

    return _grad_elec(weight, ao, eri, s1, h1aos, mol.natm, tuple([tuple(a) for a in aoslices.tolist()]), mask, mo_energy, mo_coeff, mol, dm, H)  + _grad_nuc(charges, coords)

def pyscf_reference(mol_str, opts):
    from pyscf import __config__
    __config__.dft_rks_RKS_grids_level = opts.level

    mol = build_mol(mol_str, opts.basis)
    mol.max_cycle = 50 
    mf = pyscf.scf.RKS(mol)
    mf.max_cycle = 50 
    mf.xc = "b3lyp" 
    mf.diis_space = 8
    pyscf_energies = []
    pyscf_hlgaps = []
    lumo         = mol.nelectron//2
    homo         = lumo - 1
    def callback(envs):
        pyscf_energies.append(envs["e_tot"]*HARTREE_TO_EV)
        hl_gap_hartree = np.abs(envs["mo_energy"][homo] - envs["mo_energy"][lumo]) * HARTREE_TO_EV
        pyscf_hlgaps.append(hl_gap_hartree)
        print("\rPYSCF: ", pyscf_energies[-1] , end="")
    mf.callback = callback
    mf.kernel()
    print("")
    forces = mf.nuc_grad_method().kernel()
    return np.array(pyscf_energies), np.array(pyscf_hlgaps), np.array(forces)

def print_difference(nanoDFT_E, nanoDFT_forces, nanoDFT_logged_E, nanoDFT_hlgap, pyscf_E, pyscf_forces, pyscf_hlgap):
    #TODO(HH): rename to match caller variable names
    nanoDFT_E = nanoDFT_E*HARTREE_TO_EV
    print("pyscf:\t\t%15f"%pyscf_E[-1])
    print("us:\t\t%15f"%nanoDFT_E)
    print("diff:\t\t%15f"%np.abs(pyscf_E[-1]-nanoDFT_E))
    print("chemAcc: \t%15f"%0.043)
    print("chemAcc/diff: \t%15f"%(0.043/np.abs(pyscf_E[-1]-nanoDFT_E)))
    print("")

    # Forces
    print()
    print("np.max(|nanoDFT_F-PySCF_F|):", np.max(np.abs(nanoDFT_forces-pyscf_forces)))

    norm_X = np.linalg.norm(nanoDFT_forces, axis=1)
    norm_Y = np.linalg.norm(pyscf_forces, axis=1)
    dot_products = np.sum(nanoDFT_forces * pyscf_forces, axis=1)
    cosine_similarity = dot_products / (norm_X * norm_Y)
    print("Force cosine similarity:",cosine_similarity)

def build_mol(mol_str, basis_name):
    mol = pyscf.gto.mole.Mole()
    mol.build(atom=mol_str, unit="Angstrom", basis=basis_name, spin=0, verbose=0)
    return mol

def reference(mol_str, opts):
    import pickle 
    import hashlib 
    filename = "precomputed/%s.pkl"%hashlib.sha256((str(mol_str) + str(opts.basis) + str(opts.level)).encode('utf-8')).hexdigest()
    print(filename)
    if not os.path.exists(filename):
        pyscf_E, pyscf_hlgap, pyscf_forces = pyscf_reference(mol_str, opts)
        with open(filename, "wb") as file: 
            pickle.dump([pyscf_E, pyscf_hlgap, pyscf_forces], file)
    else: 
        pyscf_E, pyscf_hlgap, pyscf_forces = pickle.load(open(filename, "rb"))
    return pyscf_E, pyscf_hlgap, pyscf_forces


if __name__ == "__main__":
    #jax.config.FLAGS.jax_platform_name = 'cpu'
    import os
    import argparse 

    parser = argparse.ArgumentParser()
    # DFT options 
    parser.add_argument('-basis',   type=str,   default="sto3g") 
    parser.add_argument('-level',   type=int,   default=0)
    # GD options 
    parser.add_argument('-backend', type=str,   default="cpu") 
    parser.add_argument('-lr',      type=float, default=1e-3)
    parser.add_argument('-steps',   type=int,   default=200)
    parser.add_argument('-bs',      type=int,   default=2)

    parser.add_argument('-normal',     action="store_true") 
    parser.add_argument('-visualize',  action="store_true") 
    opts = parser.parse_args()

    # benzene 
    if True: 
        mol_str = [
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
    else: 
        mol_str = [
            ["N",  (-1.3289  ,   1.0488 ,   -1.5596)],
            ["C",  ( 0.1286  ,   1.0198 ,   -1.8261)],
            ["C",  ( 0.3335  ,   0.8585 ,   -3.3268)],
            ["O",  (-0.0551  ,  -0.0282 ,   -4.0649)],
            ["O",  ( 1.0668  ,   1.8338 ,   -3.9108)],
            ["C",  ( 0.8906  ,  -0.1043 ,   -1.0999)],
            ["H",  ( 1.9534  ,  -0.0888 ,   -1.4126)],
            ["H",  ( 0.4975  ,  -1.0987 ,   -1.3971)],
            ["C",  ( 0.8078  ,   0.0465 ,    0.3677)],
            ["C",  ( 1.5802  ,   0.8809 ,    1.1516)],
            ["N",  ( 1.1567  ,   0.7746 ,    2.4944)],
            ["H",  ( 1.7094  ,   1.0499 ,    3.2650)],
            ["C",  ( 0.1694  ,  -0.2350 ,    2.5662)],
            ["C",  (-0.0897  ,  -0.6721 ,    1.2403)],
            ["C",  (-1.0740  ,  -1.6418 ,    1.0106)],
            ["H",  (-1.2812  ,  -1.9849 ,   -0.0088)],
            ["C",  (-1.7623  ,  -2.1470 ,    2.0948)],
            ["H",  (-2.5346  ,  -2.9080 ,    1.9416)],
            ["C",  (-1.4948  ,  -1.7069 ,    3.4060)],
            ["H",  (-2.0660  ,  -2.1385 ,    4.2348)],
            ["C",  (-0.5337  ,  -0.7507 ,    3.6638)],
            ["H",  (-0.3249  ,  -0.4086 ,    4.6819)],
            ["H",  ( 2.3719  ,   1.5631 ,    0.8380)],
            ["H",  (-1.4726  ,   1.2086 ,   -0.5841)],
            ["H",  (-1.7404  ,   0.1740 ,   -1.8129)],
            ["H",  ( 0.5299  ,   2.0096 ,   -1.4901)],
            ["H",  ( 1.1361  ,   1.6737 ,   -4.8470)],
        ]


    #pos = [np.array(a[1]).reshape(1, 1) for a in mol_str]
    #distances = map(lambda x: np.linalg.norm(np.array(x[0]) - np.array(x[1])), combinations(coords, 2))
    #return min(distances)




    mol = build_mol(mol_str, opts.basis)
    ic(mol.nao_nr())
    ic(mol.nelectron)

    pyscf_E, pyscf_hlgap, pyscf_forces = reference(mol_str, opts)
    
    nanoDFT_E, (nanoDFT_hlgap, mo_energy, mo_coeff, grid_coords, grid_weights, dm, H) = nanoDFT(mol_str, opts, pyscf_E)
    nanoDFT_forces = grad(mol, grid_coords, grid_weights, mo_coeff, mo_energy, np.array(dm), np.array(H))

    print_difference(nanoDFT_E, nanoDFT_forces, 0 , nanoDFT_hlgap, pyscf_E, pyscf_forces, pyscf_hlgap)
