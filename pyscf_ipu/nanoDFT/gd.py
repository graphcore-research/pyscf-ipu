# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
# (assumes newest Jax)
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pyscf
import optax
from icecream import ic
from pyscf_ipu.exchange_correlation.b3lyp import b3lyp, vxc_b3lyp
from tqdm import tqdm 

HARTREE_TO_EV, EPSILON_B3LYP, HYB_B3LYP = 27.2114079527, 1e-20, 0.2
def orth(x): return jnp.linalg.qr(x)[0]

def dm_energy(weights: NxK, state):
    eigvects: NxK       = state.L_inv.T @ orth(weights)                           
    density_matrix: NxN = 2 * eigvects @ eigvects.T 
    E_xc: float         = exchange_correlation(density_matrix, state.grid_AO, state.grid_weights)              
    diff_JK: NxN        = get_JK(density_matrix, state.ERI)                
    energy: float       = jnp.sum(density_matrix * (state.H_core + diff_JK/2)) + E_xc + state.E_nuc
    return energy, density_matrix

def exchange_correlation(density_matrix: NxN, grid_AO: _4xGsizexN, grid_weights: gsize):
    grid_AO_dm: _1xGsizexN = jnp.expand_dims(grid_AO[0] @ density_matrix)    # O(gsize N^2) flops and gsizeN reads.                                                                               
    mult: _4xGsizexN       = grid_AO_dm * grid_AO 
    rho: _4xGsize          = jnp.sum(mult, axis=2)                
    E_xc: Gsize            = b3lyp(rho, EPSILON_B3LYP)                                              
    E_xc: float            = jnp.sum(rho[0] * grid_weights * E_xc)
    return E_xc 

def get_JK(density_matrix: NxN, ERI: NxNxNxN):
    J: (N, N) = jnp.einsum('ijkl,ji->kl', ERI, density_matrix) 
    K: (N, N) = jnp.einsum('ijkl,jk->il', ERI, density_matrix)
    return J - (K / 2 * HYB_B3LYP)

def nanoDFT(mol_str, opts, pyscf_E):
    # Init DFT tensors on CPU using PySCF.
    mol = build_mol(mol_str, opts.basis)
    pyscf_E, pyscf_hlgap, pycsf_forces = reference(mol_str, opts)

    N = mol.nao_nr()
    state = init_dft(mol, opts)[0]
    target = pyscf_E[-1] 

    w = np.eye(N) + np.random.normal(0, 0.01, (N, N))

    vandg = jax.jit(jax.value_and_grad( dm_energy, has_aux=True), backend=opts.backend)

    # Build initializers for params
    adam = optax.adam(opts.lr)
    adam_state = adam.init(w)

    pbar = tqdm(range(opts.steps))
    for i in pbar:
        (val, density_matrix), grad = vandg(w, state)
        updates, adam_state = adam.update(grad, adam_state)
        w =  optax.apply_updates(w, updates)
        pbar.set_description("energy=%.7f [eV] error=%.7f [eV]"%(val*HARTREE_TO_EV, target-val*HARTREE_TO_EV))
        if i == 0: print("")

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
    E_nuc: np.array
    density_matrix: np.array 
    kinetic: np.array
    nuclear: np.array
    O: np.array
    mask: np.array
    L_inv: np.array
    L: np.array
    H_core: np.array
    grid_AO: np.array
    grid_weights: np.array
    atom_pos: np.array
    ERI: np.array
    grid_coords: np.array

def init_dft(mol, opts):
    N                = mol.nao_nr()                                 
    n_electrons_half = mol.nelectron//2                             
    E_nuc            = mol.energy_nuc()                             

    from pyscf import dft
    grids            = pyscf.dft.gen_grid.Grids(mol)
    grids.level      = opts.level
    grids.build()
    grid_weights    = grids.weights                                 
    coord_str       = 'GTOval_cart_deriv1' if mol.cart else 'GTOval_sph_deriv1'
    grid_AO         = mol.eval_gto(coord_str, grids.coords, 4)      
    grid_coords = grids.coords
    density_matrix  = pyscf.scf.hf.init_guess_by_minao(mol)         

    # TODO(): Add integral math formulas for kinetic/nuclear/O/ERI.
    kinetic         = mol.intor_symmetric('int1e_kin')             
    nuclear         = mol.intor_symmetric('int1e_nuc')             
    O               = mol.intor_symmetric('int1e_ovlp')            
    L = np.linalg.cholesky(O)
    L_inv           = np.linalg.inv(L)          

    mask = np.concatenate([np.ones(n_electrons_half), np.zeros(N-n_electrons_half)])

    ERI = mol.intor("int2e_sph")
    
    state = IterationState(E_nuc=E_nuc, ERI=ERI,  grid_coords=grid_coords,
                           density_matrix=density_matrix, kinetic=kinetic,
                           nuclear=nuclear, 
                           O=O, 
                           mask=mask, 
                           H_core=nuclear+kinetic,
                           L_inv=L_inv, L=L, grid_AO=grid_AO, grid_weights=grid_weights, atom_pos=mol.atom_coords())

    print("DFT Tensor Summary")
    for field_name, field_def in state.__dataclass_fields__.items():
        field_value = getattr(state, field_name)
        print(f"{field_name}: {getattr(field_value, 'shape', None)}")

    return state, n_electrons_half, E_nuc, N, L_inv, grid_weights, grid_coords, grid_AO


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
    import os 
    os.makedirs("precomputed", exist_ok=True)
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
    opts = parser.parse_args()

    # benzene 
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


    mol = build_mol(mol_str, opts.basis)
    
    ic(mol.nao_nr())
    ic(mol.nelectron)

    pyscf_E, pyscf_hlgap, pyscf_forces = reference(mol_str, opts)
    
    nanoDFT_E, (nanoDFT_hlgap, mo_energy, mo_coeff, grid_coords, grid_weights, dm, H) = nanoDFT(mol_str, opts, pyscf_E)
    nanoDFT_forces = grad(mol, grid_coords, grid_weights, mo_coeff, mo_energy, np.array(dm), np.array(H))

    print_difference(nanoDFT_E, nanoDFT_forces, 0 , nanoDFT_hlgap, pyscf_E, pyscf_forces, pyscf_hlgap)
