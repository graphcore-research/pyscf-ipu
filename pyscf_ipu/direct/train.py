# todo: use int8 instead of int64 for nonzero_indices (N<=256 so ijkl in int8 is ok). 
import os
os.environ['OMP_NUM_THREADS'] = '29'
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import scipy 
import numpy as np
import pyscf
import optax
from icecream import ic
from exchange_correlation.b3lyp import _b3lyp, vxc_b3lyp
from tqdm import tqdm 
import time 
from transformer import transformer, transformer_init
import pandas as pd 
import math 
from functools import partial 
import pickle 
import random 
from sparse_symmetric_ERI import sparse_symmetric_einsum
random.seed(42)
np.random.seed(42)

unit = "bohr"
MD17_WATER, MD17_ETHANOL, MD17_ALDEHYDE, MD17_URACIL = 1, 2, 3, 4
cfg, HARTREE_TO_EV, EPSILON_B3LYP, HYB_B3LYP = None, 27.2114079527, 1e-20, 0.2
angstrom_to_bohr, bohr_to_angstrom, nm_to_bohr, bohr_to_nm = 1.8897, 0.529177, 18.8973, 0.0529177249
B, BxNxN, BxNxK = None, None, None

# Only need to recompute: L_inv, grid_AO, grid_weights, H_core, ERI and E_nuc. 
def dm_energy(W: BxNxK, state, normal, nn, cfg=None, opts=None): 
    if opts.nn_f32: state.pos, state.H_core, state.L_inv = state.pos.astype(jnp.float32), state.H_core.astype(jnp.float32), state.L_inv.astype(jnp.float32)

    if not opts.loss_vxc: # use loss without v_xc. 
        W = jax.vmap(transformer, in_axes=(None, None, 0, 0, 0, 0,0), out_axes=(0))(cfg, \
            W, state.ao_types, state.pos, state.H_core, state.L_inv, state).astype(jnp.float64)
        L_inv_Q: BxNxN        = state.L_inv_T @ jnp.linalg.eigh(state.L_inv @ (state.H_core + W) @ state.L_inv_T)[1]      # O(B*N*K^2) FLOP O(B*N*K) FLOP/FLIO
        density_matrix: BxNxN = 2 * (L_inv_Q*state.mask) @ jnp.transpose(L_inv_Q, (0,2,1))                            # O(B*N*K^2) FLOP/FLIO 
        pred_density_matrix = density_matrix
        E_xc: B               = exchange_correlation(density_matrix, state.grid_AO, state.grid_weights, normal, opts.xc_f32)     # O(B*gsize*N^2) FLOP O(gsize*N^2) FLIO
        diff_JK: BxNxN        = JK(density_matrix, state, normal, opts.foriloop, opts.eri_f32, opts.bs)        # O(B*num_ERIs) FLOP O(num_ERIs) FLIO
        H = jnp.zeros(density_matrix.shape) 
        additional_loss  = 0
    else: 
        # turn initial dft guess into hamiltonian guess. 
        density_matrix = state.dm_init
        N = density_matrix.shape[1]
        diff_JK: BxNxN = JK(density_matrix, state, normal, opts.foriloop, opts.eri_f32, opts.bs)        
        E_xc, V_xc     = explicit_exchange_correlation( density_matrix[0], state.grid_AO[0], state.grid_weights[0])
        H_init         = state.H_core + diff_JK + V_xc  

        # predict neural network correction to initial hamiltonian guess. 
        W = jax.vmap(transformer, in_axes=(None, None, 0, 0, 0, 0,0,0,0,0), out_axes=(0))(cfg, \
            W, state.ao_types, state.pos, state.H_core, state.L_inv, state.dm_init, diff_JK, V_xc.reshape(1, N,N), H_init).astype(jnp.float64)
        H = H_init + W 

        # Given H compute new (dm, H) 
        L_inv_Q        = state.L_inv_T @ jnp.linalg.eigh(state.L_inv @ H @ state.L_inv_T)[1]   
        density_matrix = 2 * (L_inv_Q*state.mask) @ jnp.transpose(L_inv_Q, (0,2,1)) 
        diff_JK: BxNxN = JK(density_matrix, state, normal, opts.foriloop, opts.eri_f32, opts.bs)        
        E_xc, V_xc     = explicit_exchange_correlation( density_matrix[0], state.grid_AO[0], state.grid_weights[0])
        H              = state.H_core + diff_JK + V_xc  

        # Do density mixing (aka residual connection). Seem to stabalize gradients. 
        L_inv_Q        = state.L_inv_T @  jnp.linalg.eigh(state.L_inv @ H @ state.L_inv_T)[1]
        density_matrix = (density_matrix + 2 * (L_inv_Q*state.mask) @ jnp.transpose(L_inv_Q, (0,2,1)))/2 # <-- density mixing 
        diff_JK: BxNxN = JK(density_matrix, state, normal, opts.foriloop, opts.eri_f32, opts.bs) 
        E_xc, V_xc     = explicit_exchange_correlation( density_matrix[0], state.grid_AO[0], state.grid_weights[0])
        H              = state.H_core + diff_JK + V_xc  

        # Density mixing seem to break energy computation (but not hamiltonian error). 
        # Doing an update without density mixing seem to fix it. 
        L_inv_Q        = state.L_inv_T @ jnp.linalg.eigh(state.L_inv @ H @ state.L_inv_T)[1]   
        density_matrix = 2 * (L_inv_Q*state.mask) @ jnp.transpose(L_inv_Q, (0,2,1)) 
        diff_JK: BxNxN = JK(density_matrix, state, normal, opts.foriloop, opts.eri_f32, opts.bs)        
        E_xc, V_xc     = explicit_exchange_correlation( density_matrix[0], state.grid_AO[0], state.grid_weights[0])
        H              = state.H_core + diff_JK + V_xc  

    energies: B           = E_xc + state.E_nuc + jnp.sum((density_matrix * (state.H_core + diff_JK/2)).reshape(W.shape[0], -1), axis=-1) 
    energy: float         = jnp.sum(energies)  
    loss = energy 
    return loss, (energies, (energy, 0, 0, 0, 0), E_xc, density_matrix, W, H)

from exchange_correlation.b3lyp import vxc_b3lyp as b3lyp
def explicit_exchange_correlation(density_matrix, grid_AO, grid_weights):
    """Compute exchange correlation integral using atomic orbitals (AO) evalauted on a grid. """
    grid_AO_dm = grid_AO[0] @ density_matrix                                                    # (gsize, N) @ (N, N) -> (gsize, N)
    grid_AO_dm = jnp.expand_dims(grid_AO_dm, axis=0)                                            # (1, gsize, N)
    mult = grid_AO_dm * grid_AO  
    rho = jnp.sum(mult, axis=2)                                                                 # (4, grid_size)=(4, 45624) for C6H6.
    E_xc, vrho, vgamma = b3lyp(rho, EPSILON_B3LYP)                                              # (gridsize,) (gridsize,) (gridsize,)
    E_xc = jnp.sum(rho[0] * grid_weights * E_xc)                                                # float=-27.968[Ha] for C6H6 at convergence.
    rho = jnp.concatenate([vrho.reshape(1, -1)/2, 4*vgamma*rho[1:4]], axis=0) * grid_weights    # (4, grid_size)=(4, 45624)
    grid_AO_T = grid_AO[0].T                                                                    # (N, gsize)
    rho = jnp.expand_dims(rho, axis=2)                                                          # (4, gsize, 1)
    grid_AO_rho = grid_AO * rho                                                                 # (4, gsize, N)
    sum_grid_AO_rho = jnp.sum(grid_AO_rho, axis=0)                                              # (gsize, N)
    V_xc = grid_AO_T @ sum_grid_AO_rho                                                          # (N, N)
    V_xc = V_xc + V_xc.T                                                                        # (N, N)
    return E_xc, V_xc   

def exchange_correlation(density_matrix: BxNxN, grid_AO, grid_weights, normal, xc_f32):
    if grid_AO.ndim == 3: 
        grid_AO = grid_AO[jnp.newaxis, ...]
        grid_weights = grid_weights[jnp.newaxis, ...]
        density_matrix = density_matrix[jnp.newaxis, ...]
    _, _, gsize, N = grid_AO.shape
    B = density_matrix.shape[0]
    if xc_f32: density_matrix = density_matrix.astype(jnp.float32)
    grid_AO_dm = (grid_AO[:, 0] @ density_matrix)                   # (B,gsize,N) @ (B, N, N) = O(B gsize N^2)
    rho        = jnp.sum(grid_AO_dm.reshape(B, 1, gsize, N) * grid_AO, axis=3).astype(jnp.float64)    # (B,1,gsize,N) * (B,4,gsize,N) = O(B gsize N)
    E_xc       = jax.vmap(_b3lyp, in_axes=(0, None))(rho, EPSILON_B3LYP).reshape(B, gsize)
    E_xc       = jnp.sum(rho[:, 0] * grid_weights * E_xc, axis=-1).reshape(B)
    return E_xc 

def JK(density_matrix, state, normal, jax_foriloop, eri_f32, bs): 
    if eri_f32: density_matrix = density_matrix.astype(jnp.float32)
    diff_JK: BxNxN = jax.vmap(sparse_symmetric_einsum, in_axes=(None, None, 0, None))(
        state.nonzero_distinct_ERI[0], 
        state.nonzero_indices[0], 
        density_matrix, 
        jax_foriloop
    )
    if bs == 1: return diff_JK
    else: return diff_JK - jax.vmap(sparse_symmetric_einsum, in_axes=(0, None, 0, None))(\
        state.diffs_ERI, 
        state.indxs, 
        density_matrix, 
        jax_foriloop).astype(jnp.float64)

def nao(atom, basis):
    m = pyscf.gto.Mole(atom='%s 0 0 0; %s 0 0 1;'%(atom, atom), basis=basis, unit=unit)
    m.build()
    return m.nao_nr()//2

def batched_state(mol_str, opts, bs, wiggle_num=0, 
                  do_pyscf=True, validation=False, 
                  extrapolate=False,
                  pad_electrons=45, 
                  pad_diff_ERIs=50000,
                  pad_distinct_ERIs=120000,
                  pad_grid_AO=25000,
                  pad_nonzero_distinct_ERI=200000,
                  pad_sparse_diff_grid=200000, 
                  mol_idx=42,
                  train=True
                  ): 
    # Set seed to ensure different rotation; initially all workers did same rotation! 
    np.random.seed(mol_idx)

    start_time = time.time()
    do_print = opts.do_print 
    if do_print: print("\t[%.4fs] start of 'batched_state'. "%(time.time()-start_time))
    max_pad_electrons, max_pad_diff_ERIs, max_pad_distinct_ERIs, max_pad_grid_AO, max_pad_nonzero_distinct_ERI, max_pad_sparse_diff_grid = \
        -1, -1, -1, -1, -1, -1

    if opts.alanine: 
        pad_electrons = 70
        padding_estimate = [210745, 219043, 18084, 193830, 1105268]
        padding_estimate = [int(a*1.1) for a in padding_estimate]
        pad_diff_ERIs, pad_distinct_ERIs, pad_grid_AO, pad_nonzero_distinct_ERI, pad_sparse_diff_grid = [int(a*8/opts.eri_bs) for a in padding_estimate]

    if opts.md17 > 0: 
        if opts.md17 == MD17_WATER:
            if opts.level == 1: padding_estimate = [    3361,  5024, 10172 , 5024  , 155958]
            if opts.level == 2: padding_estimate = [    -1,  5024, -1 , 5024  ,  -1]
            if opts.level == 3: padding_estimate = [    3361,  5024, 34310 , 5024  ,  600000]
            #if opts.level == 3: padding_estimate = [    -1,  5024, -1 , 5024  ,  -1]
            if opts.bs == 2 and opts.wiggle_var == 0: padding_estimate = [    1,  5024, 10172 , 5024    , 1]
            padding_estimate = [int(a*1.5) for a in padding_estimate]
            pad_diff_ERIs, pad_distinct_ERIs, pad_grid_AO, pad_nonzero_distinct_ERI, pad_sparse_diff_grid = padding_estimate

        elif opts.md17 == MD17_ETHANOL:
            pad_electrons = 72
            if opts.level == 0: padding_estimate = [ 98733, 166739, -1, 121315, -1]
            if opts.level == 1: padding_estimate = [ 98733, 125853, -1, 121315, -1]
            if opts.level == 2: padding_estimate = [293198, 350007, -1, 348561, -1] 
            #if opts.level == 3: padding_estimate = [151411, 350007, -1, 348561, -1]
            if opts.level == 3: padding_estimate = [-1, 422402, -1, 418726, -1]

            if opts.bs == 1 and opts.level == 2: 
                padding_estimate = [    -1, 364817    , -1 ,364701     ,-1]

            padding_estimate = [int(a*1.20) for a in padding_estimate]
            pad_diff_ERIs, pad_distinct_ERIs, pad_grid_AO, pad_nonzero_distinct_ERI, pad_sparse_diff_grid = padding_estimate
            atom1, atom2 = np.random.choice(range(1,7), size=2, replace=False).tolist() # move two random hydrogens 
            atom3 = np.random.choice(range(0,3), size=1, replace=False).tolist()[0] #  move one heavy atom 

        elif opts.md17 == MD17_ALDEHYDE:
            pad_electrons = 90 
            #if opts.level == 1: padding_estimate = [ 582957 ,962228   ,35704  ,961321 ,1309445] 
            #if opts.level == 1: padding_estimate = [190496 ,611582  ,35704 ,607874 ,691612] 
            if opts.level == 1: padding_estimate = [    -1, 711397  ,   -1, 710933   ,  -1]
            if opts.level == 2: padding_estimate = [    -1, 711397 *2 ,   -1, 710933  *2 ,  -1]
            padding_estimate = [int(a*1.2) for a in padding_estimate]
            pad_diff_ERIs, pad_distinct_ERIs, pad_grid_AO, pad_nonzero_distinct_ERI, pad_sparse_diff_grid = padding_estimate

        elif opts.md17 == MD17_URACIL: 
            pad_electrons = 132
            if opts.level == 2: padding_estimate = [     -1, 2677777     , -1, 2676125    ,  -1] 
            if opts.level == 3: padding_estimate = [     -1, 3620179     , -1, 3620179    ,  -1] 
            padding_estimate = [int(a*1.20) for a in padding_estimate]
            pad_diff_ERIs, pad_distinct_ERIs, pad_grid_AO, pad_nonzero_distinct_ERI, pad_sparse_diff_grid = padding_estimate

    mol = build_mol(mol_str, opts.basis)
    pad_electrons = min(pad_electrons, mol.nao_nr())
        
    if opts.alanine:
        # train on [-180, 180], validate [-180, 180] extrapolate [-360, 360]\[180, -180]
        if extrapolate: phi, psi = [float(a) for a in np.random.uniform(180, 360, 2)]
        else: phi, psi = [float(a) for a in np.random.uniform(0, 180, 2)]
        angles = []

    if opts.md17 > 0: 
        natm = len(mol_str)
        atom_num = random.sample(range(natm), 1)[0]
        atoms = np.array([mol_str[i][1] for i in range(0,natm)])
        if opts.bs > 1: 
            if opts.md17 == MD17_WATER: from _ase import get_md_trajectory_water as md_traj 
            else: from _ase import get_md_trajectory as md_traj 
            atoms = "".join([a[0] for a in mol_str])
            positions = np.array([a[1] for a in mol_str])*bohr_to_angstrom
            positions = md_traj(atoms, positions, steps=bs, dt=opts.md_time, temperature=opts.md_T)
            positions = [a*angstrom_to_bohr for a in positions]

    states = []
    import copy 
    for iteration in range(bs):
        new_str = copy.deepcopy(mol_str)

        if do_print: print("\t[%.4fs] initializing state %i. "%(time.time()-start_time, iteration))
        if opts.alanine: 
            from rdkit import Chem
            from rdkit.Chem import AllChem
            pdb_file = 'alanine.pdb'
            molecule = Chem.MolFromPDBFile(pdb_file, removeHs=False)
            # tried reading from pdb_block; caused parallel dataloader pickle to break. 
            AllChem.EmbedMolecule(molecule)
            #AllChem.UFFOptimizeMolecule(molecule)
            phi_atoms = [4, 6, 8, 14]  # indices for phi dihedral
            psi_atoms = [6, 8, 14, 16]  # indices for psi dihedral

            def xyz(atom): return np.array([atom.x, atom.y, atom.z]).reshape(1, 3)
            def get_atom_positions(mol):
                conf = mol.GetConformer()
                return np.concatenate([xyz(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], axis=0)

            str = [new_str[j][0] for j in range(len(new_str))]
            pos = np.concatenate([np.array(new_str[j][1]).reshape(1, 3) for j in range(len(new_str))])

            AllChem.SetDihedralDeg(molecule.GetConformer(), *phi_atoms, phi)
            angle = psi + float(np.random.uniform(0, opts.rotate_deg, 1)) 

            if extrapolate:  angle = angle % 180 + 180 
            else: angle = angle % 180 

            AllChem.SetDihedralDeg(molecule.GetConformer(), *psi_atoms, angle )
            pos = get_atom_positions(molecule)
            angles.append((phi, angle))

            for j in range(len(new_str)): new_str[j][1] = tuple(pos[j])

        if opts.md17 > 0:
            if iteration > 0:
                import copy 
                #new_str[atom_num][1] = tuple(atoms[atom_num] + np.random.normal(0,opts.wiggle_var, (3)))
                pos = positions[iteration]
                if opts.md17 == MD17_WATER: 
                    new_str[0][1] = tuple(pos[0])
                    new_str[1][1] = tuple(pos[1])
                    new_str[2][1] = tuple(pos[2])
                elif opts.md17 == MD17_ETHANOL: # ethanol =  c2h6o 
                    new_str[-atom1][1] = tuple(pos[-atom1]) 
                    new_str[-atom2][1] = tuple(pos[-atom2])
                    new_str[atom3][1] = tuple(pos[atom3])
                elif opts.md17 == MD17_ALDEHYDE: # malondialdehyde = c3h4o2, ethanol =  c2h6o 
                    new_str[-1][1] = tuple(pos[-1]) 
                    new_str[-3][1] = tuple(pos[-3])

        if iteration == 0: 
            state = init_dft(new_str, opts, do_pyscf=do_pyscf, pad_electrons=pad_electrons)
            c, w = state.grid_coords, state.grid_weights
        elif iteration <= 1 or not opts.prof:  # when profiling create fake molecule to skip waiting
            state = init_dft(new_str, opts, c, w, do_pyscf=do_pyscf and iteration < 3, state=state, pad_electrons=pad_electrons)
        states.append(state)

    state = cats(states)
    N = state.N[0]
    if do_print: print("\t[%.4fs] concatenated states. "%(time.time()-start_time))

    # Compute ERI sparsity. 
    nonzero = []
    for e, i in zip(state.nonzero_distinct_ERI, state.nonzero_indices):
        abs = np.abs(e)
        indxs = abs < opts.eri_threshold #1e-10 
        e[indxs] = 0 
        nonzero.append(np.nonzero(e)[0])

    if do_print: print("\t[%.4fs] got sparsity. "%(time.time()-start_time))

    # Merge nonzero indices and prepare (ij, kl).
    # rep is the number of repetitions we include in the sparse representation. 
    union = nonzero[0]
    for i in range(1, len(nonzero)): 
        union = np.union1d(union, nonzero[i])
    nonzero_indices = union 
    if do_print: print("\t[%.4fs] got union of sparsity. "%(time.time()-start_time))

    from sparse_symmetric_ERI import get_i_j, num_repetitions_fast
    ij, kl               = get_i_j(nonzero_indices)
    rep                  = num_repetitions_fast(ij, kl)
    if do_print: print("\t[%.4fs] got (ij) and reps. "%(time.time()-start_time))

    batches = opts.eri_bs
    es = []
    for e,i in zip(state.nonzero_distinct_ERI, state.nonzero_indices):
        nonzero_distinct_ERI = e[nonzero_indices] / rep
        remainder            = nonzero_indices.shape[0] % (batches)
        if remainder != 0: nonzero_distinct_ERI = np.pad(nonzero_distinct_ERI, (0,batches-remainder))

        nonzero_distinct_ERI = nonzero_distinct_ERI.reshape(batches, -1)
        es.append(nonzero_distinct_ERI)

    state.nonzero_distinct_ERI = np.concatenate([np.expand_dims(a, axis=0) for a in es])

    if do_print: print("\t[%.4fs] padded ERI and nonzero_indices. . "%(time.time()-start_time))
    i, j = get_i_j(ij.reshape(-1))
    k, l = get_i_j(kl.reshape(-1))
    if do_print: print("\t[%.4fs] got ijkl. "%(time.time()-start_time))

    if remainder != 0:
        i = np.pad(i, ((0,batches-remainder)))
        j = np.pad(j, ((0,batches-remainder)))
        k = np.pad(k, ((0,batches-remainder)))
        l = np.pad(l, ((0,batches-remainder)))
    nonzero_indices = np.vstack([i,j,k,l]).T.reshape(batches, -1, 4).astype(np.int32) # todo: we can use int8 here. 
    state.nonzero_indices = nonzero_indices  
    if do_print: print("\t[%.4fs] padded and vstacked ijkl. "%(time.time()-start_time))


    if opts.normal: diff_state = None 
    else: 
        # use the same sparsity pattern across a batch.
        if opts.bs > 1: 
            diff_ERIs  = state.nonzero_distinct_ERI[:1] - state.nonzero_distinct_ERI
            diff_indxs = state.nonzero_indices.reshape(1, batches, -1, 4)
            nzr        = np.abs(diff_ERIs[1]).reshape(batches, -1) > 1e-10

            diff_ERIs  = diff_ERIs[:, nzr].reshape(bs, -1)
            diff_indxs = diff_indxs[:, nzr].reshape(-1, 4)

            remainder = np.sum(nzr) % batches
            if remainder != 0:
                diff_ERIs = np.pad(diff_ERIs, ((0,0),(0,batches-remainder)))
                diff_indxs = np.pad(diff_indxs, ((0,batches-remainder),(0,0)))

            diff_ERIs = diff_ERIs.reshape(bs, batches, -1)
            diff_indxs = diff_indxs.reshape(batches, -1, 4)


            if pad_diff_ERIs == -1: 
                state.indxs=diff_indxs
                state.diffs_ERI=diff_ERIs
                assert False, "deal with precomputed_indxs; only added in else branch below"
            else: 
                max_pad_diff_ERIs = diff_ERIs.shape[2]
                if do_print: print("\t[%.4fs] max_pad_diff_ERIs=%i"%(time.time()-start_time, max_pad_diff_ERIs))
                # pad ERIs with 0 and indices with -1 so they point to 0. 
                assert diff_indxs.shape[1] == diff_ERIs.shape[2]
                pad = pad_diff_ERIs - diff_indxs.shape[1]
                assert pad > 0, (pad_diff_ERIs, diff_indxs.shape[1])
                state.indxs     = np.pad(diff_indxs, ((0,0), (0, pad), (0, 0)), 'constant', constant_values=(-1))
                state.diffs_ERI = np.pad(diff_ERIs,  ((0,0), (0, 0),   (0, pad))) # pad zeros 

        #state.grid_AO = state.grid_AO[:1]
        state.nonzero_distinct_ERI = state.nonzero_distinct_ERI[:1]
        state.nonzero_indices = np.expand_dims(state.nonzero_indices, axis=0)

        # todo: looks like we're padding, then looking for zeros, then padding; this can be simplified. 
        if pad_distinct_ERIs != -1: 
            max_pad_distinct_ERIs = state.nonzero_distinct_ERI.shape[2]
            if do_print: print("\t[%.4fs] max_pad_distinct_ERIs=%i"%(time.time()-start_time, max_pad_diff_ERIs))
            assert state.nonzero_distinct_ERI.shape[2] == state.nonzero_indices.shape[2]
            pad = pad_distinct_ERIs - state.nonzero_distinct_ERI.shape[2]
            assert pad > 0, (pad_distinct_ERIs, state.nonzero_distinct_ERI.shape[2])
            state.nonzero_indices      = np.pad(state.nonzero_indices,      ((0,0), (0,0), (0, pad), (0,0)), 'constant', constant_values=(-1))
            state.nonzero_distinct_ERI = np.pad(state.nonzero_distinct_ERI, ((0,0), (0,0),  (0, pad))) # pad zeros 

        indxs = np.abs(state.nonzero_distinct_ERI ) > opts.eri_threshold #1e-9 
        state.nonzero_distinct_ERI = state.nonzero_distinct_ERI[indxs]
        state.nonzero_indices = state.nonzero_indices[indxs]
        remainder = state.nonzero_indices.shape[0] % batches

        if remainder != 0:
            state.nonzero_distinct_ERI = np.pad(state.nonzero_distinct_ERI, (0,batches-remainder))
            state.nonzero_indices = np.pad(state.nonzero_indices, ((0,batches-remainder), (0,0)))
        state.nonzero_distinct_ERI = state.nonzero_distinct_ERI.reshape(1, batches, -1)
        state.nonzero_indices = state.nonzero_indices.reshape(1, batches, -1, 4)

        if pad_nonzero_distinct_ERI != -1: 
            max_pad_nonzero_distinct_ERI = state.nonzero_distinct_ERI.shape[2]
            if do_print: print("\t[%.4fs] max_pad_nonzero_distinct_ERI=%i"%(time.time()-start_time, max_pad_nonzero_distinct_ERI))

            assert state.nonzero_distinct_ERI.shape[2] == state.nonzero_indices.shape[2]
            pad = pad_nonzero_distinct_ERI - state.nonzero_distinct_ERI.shape[2]
            assert pad >= 0, (pad_nonzero_distinct_ERI, state.nonzero_distinct_ERI.shape[2])
            state.nonzero_distinct_ERI = np.pad(state.nonzero_distinct_ERI, ((0,0),(0,0),(0,pad)))
            state.nonzero_indices = np.pad(state.nonzero_indices, ((0,0),(0,0),(0,pad), (0,0)), 'constant', constant_values=(-1))

    # this may upset vmap. 
    B = state.grid_AO.shape[0]
    state.pad_sizes = np.concatenate([np.array([
        max_pad_diff_ERIs, max_pad_distinct_ERIs, max_pad_grid_AO, 
        max_pad_nonzero_distinct_ERI, max_pad_sparse_diff_grid]).reshape(1, -1) for _ in range(B)])

    if opts.eri_f32: 
        state.nonzero_distinct_ERI = state.nonzero_distinct_ERI.astype(jnp.float32)
        state.diffs_ERI = state.diffs_ERI.astype(jnp.float32)

    if opts.xc_f32: 
        state.main_grid_AO = state.main_grid_AO.astype(jnp.float32)
        state.grid_AO = state.grid_AO.astype(jnp.float32)

    

    return state 


def nanoDFT(mol_str, opts):
    start_time = time.time()
    print()
    if opts.wandb: 
        import wandb 
        if opts.alanine:
            run = wandb.init(project='ndft_alanine')
        elif opts.qm9: 
            run = wandb.init(project='ndft_qm9')
        elif opts.md17 > 0: 
            run = wandb.init(project='md17')
        else: 
            run = wandb.init(project='ndft')
        opts.name = run.name

        wandb.log(vars(opts))

    else:
        opts.name = "%i"%time.time()

    rnd_key = jax.random.PRNGKey(42)
    n_vocab = nao("C", opts.basis) + nao("N", opts.basis) + \
              nao("O", opts.basis) + nao("F", opts.basis) + \
              nao("H", opts.basis)  

    global cfg
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
    if opts.large:  # this is 600M; 
        d_model= 1280 
        n_heads = 16
        n_layers = 36
    if opts.largep:  # interpolated between large and largep. 
        d_model= 91*16 # halway from 80 to 100 
        n_heads = 16*1 # this is 1.3B; decrease parameter count 30%. 
        n_layers = 43
    if opts.xlarge:  
        d_model= 1600 
        n_heads = 25 
        n_layers = 48

    if opts.nn: 
        rnd_key, cfg, params, total_params = transformer_init(
            rnd_key,
            n_vocab,
            d_model =d_model,
            n_layers=n_layers,
            n_heads =n_heads,
            d_ff    =d_model*4,
        )
        print("[%.4fs] initialized transformer. "%(time.time()-start_time) )
        if opts.nn_f32: params = params.to_float32()

        from natsort import natsorted 
        if opts.resume: 
            all = os.listdir("checkpoints")
            candidates = natsorted([a for a in all if opts.resume in a])
            print(candidates)
            print("found candidates", candidates)

            to_load = candidates[-1].replace("_model.pickle", "").replace("_adam_state.pickle", "")
            print("choose candidate ", to_load)
            opts.resume = to_load 

        if opts.resume: 
            print("loading checkpoint")
            params = pickle.load(open("checkpoints/%s_model.pickle"%opts.resume, "rb"))
            if opts.nn_f32: params = params.to_float32()
            else:  params = params.to_float64()
            print("done loading. ")


    if opts.nn: 
        #https://arxiv.org/pdf/1706.03762.pdf see 5.3 optimizer 
        def custom_schedule(it, learning_rate=opts.lr, min_lr=opts.min_lr, warmup_iters=opts.warmup_iters, lr_decay_iters=opts.lr_decay): 
            cond1 = (it < warmup_iters) * learning_rate * it / warmup_iters
            cond2 = (it > lr_decay_iters) * min_lr
            decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
            coeff = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_ratio)) 
            cond3 = (it >= warmup_iters) * (it <= lr_decay_iters) * (min_lr + coeff * (learning_rate - min_lr))
            #if not opts.resume: return cond1 + cond2 + cond3 
            return cond1 + cond2 + cond3 
 
        adam = optax.chain(
            optax.clip_by_global_norm(1),
            optax.scale_by_adam(b1=0.9, b2=0.95, eps=1e-12),
            optax.add_decayed_weights(0.1),
            optax.scale_by_schedule(custom_schedule),
            optax.scale(-1),
            #optax.ema(opts.ema) # doesn't seem to help. 
        )
        w = params 

        df = None 
        if opts.qh9: 
            df = pd.read_pickle("qh9/qh9stable_processed_shuffled.pickle")
            df = df[df["N_sto3g"]==55] 
            print(df.shape)
        elif opts.qm9: 
            df = pd.read_pickle("alchemy/processed_atom_9.pickle") # spin=0 and only CNOFH molecules 
            if nao != -1: df = df[df["nao"]==nao] 

        print(jax.devices())

        from torch.utils.data import DataLoader, Dataset
        class OnTheFlyQM9(Dataset):
            # prepares dft tensors with pyscf "on the fly". 
            # dataloader is very keen on throwing segfaults (e.g. using jnp in dataloader throws segfaul). 
            # problem: second epoch always gives segfault. 
            # hacky fix; make __len__ = real_length*num_epochs and __getitem__ do idx%real_num_examples 
            def __init__(self, opts, df=None, nao=294, train=True, num_epochs=10**9, extrapolate=False):
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
                self.do_pyscf = self.validation or self.extrapolate

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
                self.train = train 
                self.md17 = opts.md17
                if opts.waters: self.mol_strs = [self.waters]

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
                else: 
                    self.H = [0 for _ in self.mol_strs] 
                    self.E = [0 for _ in self.mol_strs]


                if opts.alanine: self.mol_strs = mol_str

                if train: self.bs = opts.bs 
                else: self.bs = opts.val_bs

                    
            def __len__(self): return len(self.mol_strs)*self.num_epochs

            def __getitem__(self, idx):
                if self.md17 == MD17_WATER and self.train: idx = random.randint(0, 499) 
                if self.md17 == MD17_ETHANOL and self.train: idx = random.randint(0, 24999) 
                if self.md17 == MD17_ALDEHYDE and self.train: idx = random.randint(0, 24999) 
                if self.md17 == MD17_URACIL and self.train: idx = random.randint(0, 24999) 
                return batched_state(self.mol_strs[idx%len(self.mol_strs)], self.opts, self.bs, \
                    wiggle_num=0, do_pyscf=self.do_pyscf, validation=False, \
                        extrapolate=self.extrapolate, mol_idx=idx, train=self.train), self.H[idx%len(self.mol_strs)], self.E[idx%len(self.mol_strs)]

        print("[%.4fs] initialized datasets. "%(time.time()-start_time) )
        val_qm9 = OnTheFlyQM9(opts, train=False, df=df)
        print("[%.4fs] initialized datasets. "%(time.time()-start_time) )

        if opts.precompute:  
            val_state = val_qm9[0]
            ext_state = ext_qm9[0]
            exit()

        qm9 = OnTheFlyQM9(opts, train=True, df=df)
        print("[%.4fs] initialized datasets. "%(time.time()-start_time) )
        if opts.workers != 0: train_dataloader = DataLoader(qm9, batch_size=1, pin_memory=True, shuffle=False, drop_last=True, num_workers=opts.workers, prefetch_factor=2, collate_fn=lambda x: x[0])
        else:                 train_dataloader = DataLoader(qm9, batch_size=1, pin_memory=True, shuffle=False, drop_last=True, num_workers=opts.workers,  collate_fn=lambda x: x[0])
        pbar = tqdm(train_dataloader)
        print("[%.4fs] initialized dataloaders. "%(time.time()-start_time) ) 

        if opts.test_dataloader:
            t0 = time.time()
            for iteration, (state, H, E) in enumerate(pbar):
                    if iteration == 0: summary(state) 
                    print(time.time()-t0)
                    t0 = time.time()
                    print(state.pad_sizes.reshape(1, -1))

            exit()

        

    vandg = jax.jit(jax.value_and_grad(dm_energy, has_aux=True), backend=opts.backend, static_argnames=("normal", 'nn', "cfg", "opts"))
    valf = jax.jit(dm_energy, backend=opts.backend, static_argnames=("normal", 'nn', "cfg", "opts"))
    adam_state = adam.init(w)
    print("[%.4fs] jitted vandg and valf."%(time.time()-start_time) )

    if opts.resume: 
        print("loading adam state")
        adam_state = pickle.load(open("checkpoints/%s_adam_state.pickle"%opts.resume, "rb"))
        print("done")

    w, adam_state = jax.device_put(w), jax.device_put(adam_state)
    print("[%.4fs] jax.device_put(w,adam_state)."%(time.time()-start_time) )


    @partial(jax.jit, backend=opts.backend)
    def update(w, adam_state, accumulated_grad):
        if opts.grad_acc: accumulated_grad = jax.tree_map(lambda x: x / (opts.bs * opts.grad_acc), accumulated_grad)
        else: accumulated_grad = jax.tree_map(lambda x: x / opts.bs, accumulated_grad)
        updates, adam_state = adam.update(accumulated_grad, adam_state, w)
        w = optax.apply_updates(w, updates)
        return w, adam_state

    if opts.wandb: 
        if not opts.nn: total_params = -1 
        wandb.log({'total_params': total_params, 'batch_size': opts.bs, 'lr': opts.lr })

    min_val, min_dm, mins, valid_str, step, val_states, ext_state = 0, 0, np.ones(opts.bs)*1e6, "", 0, [], None
    t0, load_time, train_time, val_time, plot_time = time.time(), 0, 0, 0, 0
    accumulated_grad = None 

    paddings = []
    states   = []

    print("[%.4fs] first iteration."%(time.time()-start_time) )

    for iteration, (state, H, E) in enumerate(pbar):
        if iteration == 0: summary(state) 
        state = jax.device_put(state) 

        # Estimate max padding. 
        if iteration < 100: 
            paddings.append(state.pad_sizes.reshape(1, -1))
            _paddings = np.concatenate(paddings, axis=0)
            print(np.max(_paddings, 0))

        dct = {}
        dct["iteraton"] = iteration 

        states.append(state)
        if len(states) > opts.mol_repeats: states.pop(0)

        if opts.shuffle: random.shuffle(states)  

        load_time, t0 = time.time()-t0, time.time()

        if len(states) < 50: print(len(states), opts.name)

        for j, state in enumerate(states):
            print(". ", end="", flush=True) 
            if j == 0: _t0 =time.time()
            # energy is something like 4k; but only gradient. may be ok? 
            (val, (vals, losses, E_xc, density_matrix, _W, _)), grad = vandg(w, state, opts.normal, opts.nn, cfg, opts)
            print(",", end="", flush=True)
            if j == 0: time_step1 = time.time()-_t0

            if opts.grad_acc == 0 or len(states) < opts.mol_repeats: 
                print("#", end="", flush=True)
                w, adam_state = update(w, adam_state, grad)
            else:  
                accumulated_grad = grad if accumulated_grad is None else jax.tree_map(lambda x, y: x + y, accumulated_grad, grad)

                if (j+1) % opts.grad_acc == 0 and j > 0: # we assume opts.grad_acc divides opts.mol_repeats; prev was basically grad_acc=0 or grad_acc=mol_repeats, can now do hybrid. 
                    w, adam_state = update(w, adam_state, grad)
                    accumulated_grad = None 
                    print("#\n", end="", flush=True)

                    
            if opts.checkpoint != -1 and adam_state[1].count % opts.checkpoint == 0 and adam_state[1].count > 0:
            t0 = time.time()
            try: 
                name = opts.name.replace("-", "_")
                path_model = "checkpoints/%s_%i_model.pickle"%(name, iteration)
                path_adam = "checkpoints/%s_%i_adam_state.pickle"%(name, iteration)
                print("trying to checkpoint to %s and %s"%(path_model, path_adam))
                pickle.dump(jax.device_get(w), open(path_model, "wb"))
                pickle.dump(jax.device_get(adam_state), open(path_adam, "wb"))
                print("done!")
                print("\t-resume \"%s\""%(path_model.replace("_model.pickle", "")))
            except: 
                print("fail!")
                pass 
            print("tried saving model took %fs"%(time.time()-t0))  
            save_time, t0 = time.time()-t0, time.time()

        global_batch_size = len(states)*opts.bs
        if opts.wandb: dct["global_batch_size"] = global_batch_size
        if opts.wandb: 
            dct["energy"] = -losses[0]
            dct["norm_errvec"] = losses[1]
            dct["H_loss"] = losses[2]
            dct["E_loss"] = losses[3]
            dct["dm_loss"] = losses[4]

        train_time, t0 = time.time()-t0, time.time() 
        update_time, t0 = time.time()-t0, time.time() 

        if not opts.nn: 
            str = "error=" + "".join(["%.7f "%(vals[i]*HARTREE_TO_EV-state.pyscf_E[i]) for i in range(2)]) + " [eV]"
            str += "pyscf=%.7f us=%.7f"%(state.pyscf_E[0]/HARTREE_TO_EV, vals[0])
        else: 
            pbar.set_description("train=%.4f"%(vals[0]*HARTREE_TO_EV) + "[eV] "+ valid_str + "time=%.1f %.1f %.1f %.1f %.1f %.1f"%(load_time, time_step1, train_time, update_time, val_time, plot_time))

        if opts.wandb:
            dct["time_load"]  = load_time 
            dct["time_step1"]  = time_step1
            dct["time_train"]  = train_time
            dct["time_val"]  = val_time 
            plot_iteration = iteration % 10 == 0

            dct["train_E"] = np.abs(E*HARTREE_TO_EV)
            dct["train_E_pred"] = np.abs(vals[0]*HARTREE_TO_EV)

        step = adam_state[1].count

        plot_time, t0 = time.time()-t0, time.time() 

        if opts.nn and (iteration < 250 or iteration % 10 == 0): 
            lr = custom_schedule(step)
            dct["scheduled_lr"] = lr

            #for j, val_idx in enumerate([1, 42, 137, 400]):
            #val_idxs = [1, 42, 137, 400]
            val_idxs = [1]
            for j, val_idx in enumerate(val_idxs):
                if len(val_states) < len(val_idxs): val_states.append(jax.device_put(val_qm9[val_idx]))
                val_state, val_H, val_E = val_states[j]
                _, (valid_vals, losses, _, vdensity_matrix, vW, H) = valf(w, val_state, opts.normal, opts.nn, cfg, opts)

                if opts.wandb: 
                    dct["energy_v"] = -losses[0]
                    dct["norm_errvec_v"] = losses[1]
                    dct["H_loss_v"] = losses[2]
                    dct["E_loss_v"] = losses[3]
                    dct["dm_loss_v"] = losses[4]

                frequency = 100 
                if iteration % frequency == 0: 
                    if opts.md17 > 0: 
                        def get_S(dm):
                            import pyscf 
                            from pyscf import gto, dft
                            m = pyscf.gto.Mole(atom=val_qm9.mol_strs[val_idx], basis="def2-svp", unit=unit)
                            m.build()
                            mf = dft.RKS(m)
                            mf.xc = 'B3LYP5'
                            mf.verbose = 0 
                            mf.diis_space = 8
                            mf.conv_tol = 1e-13
                            mf.grad_tol = 3.16e-5
                            mf.grids.level = opts.level
                            #mf.kernel()
                            S = mf.get_ovlp()
                            return S 
                        S = get_S(vdensity_matrix[0])
                        def get_H_from_dm(dm):
                            import pyscf 
                            from pyscf import gto, dft
                            m = pyscf.gto.Mole(atom=val_qm9.mol_strs[val_idx], basis="def2-svp", unit=unit)
                            m.build()
                            mf = dft.RKS(m)
                            mf.xc = 'B3LYP5'
                            mf.verbose = 0 
                            mf.diis_space = 8
                            mf.conv_tol = 1e-13
                            mf.grad_tol = 3.16e-5
                            mf.grids.level = opts.level
                            #mf.kernel()
                            h_core = mf.get_hcore()
                            S = mf.get_ovlp()
                            vxc = mf.get_veff(m, dm) 
                            H = h_core + vxc 
                            S = mf.get_ovlp()
                            return H, S 


                        if not opts.loss_vxc or True:  
                            print(H.shape, vdensity_matrix.shape)
                            matrix = np.array(vdensity_matrix[0])
                            N = int(np.sqrt(matrix.size))
                            _val_H, S = get_H_from_dm(matrix.reshape(N, N))
                            print(_val_H.shape)
                            print(_val_H.reshape(-1)[:5])
                            print(H.reshape(-1)[:5])
                            print(np.max(np.abs(H[0] - _val_H)))
                            print(np.mean(np.abs(H[0] - _val_H)))
                        else: 
                            _val_H = H[0]


                        # compare eigenvalues 
                        pred_vals  = scipy.linalg.eigh(_val_H, S)[0]
                        label_vals = scipy.linalg.eigh(val_H, S)[0]
                        MAE_vals = np.mean(np.abs(pred_vals - label_vals))
                        dct["val_eps%i"%val_idx] = MAE_vals
                    dct['val_H_MAE_%i'%val_idx] = np.mean(np.abs(val_H - _val_H)) # perhaps sign doesn't matter? 

                dct['val_E_%i'%val_idx] = np.abs(valid_vals[0]*HARTREE_TO_EV-val_E*HARTREE_TO_EV )

                for i in range(0, 3):
                    dct['valid_l%i_%i'%(val_idx, i) ] = np.abs(valid_vals[i]*HARTREE_TO_EV-val_state.pyscf_E[i])
                    dct['valid_E%i_%i'%(val_idx, i) ] = np.abs(valid_vals[i]*HARTREE_TO_EV)
                    dct['valid_pyscf%i_%i'%(val_idx, i) ] = np.abs(val_state.pyscf_E[i])
            
            valid_str =  "lr=%.3e"%lr + "val=%.4f [eV] "%(valid_vals[0]*HARTREE_TO_EV-val_E*HARTREE_TO_EV)   
            if opts.md17> 0:valid_str+= " eps=%.4f"%(MAE_vals)
            valid_str +=  "val'=" + "".join(["%.4f "%(valid_vals[i]*HARTREE_TO_EV-val_state.pyscf_E[i]) for i in range(0, 3)]) + " [eV]"


            

        
        if opts.wandb: 
            dct["step"] = step 
            wandb.log(dct)
        val_time, t0 = time.time()-t0, time.time()

    val, density_matrix = min_val, min_dm

    exit()
    # needs batching 
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
    mask: np.array
    init: np.array
    E_nuc: np.array
    L_inv: np.array
    L_inv_T: np.array
    H_core: np.array
    grid_AO: np.array
    grid_weights: np.array
    grid_coords: np.array
    pyscf_E: np.array
    N: int 
    ERI: np.array
    nonzero_distinct_ERI: list 
    nonzero_indices: list
    diffs_ERI: np.array
    main_grid_AO: np.array
    diffs_grid_AO: np.array
    indxs: np.array
    sparse_diffs_grid_AO: np.array
    rows: np.array
    cols: np.array
    pos: np.array
    ao_types: np.array
    pad_sizes: np.array
    precomputed_nonzero_indices: np.array
    precomputed_indxs: np.array
    forces: np.array
    O: np.array
    dm_init: np.array

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

def gen_grid_partition(coords, atom_coords, natm, atm_dist, elements, 
                       atomic_radii,  becke_scheme=original_becke,):
    ngrids = coords.shape[0]
    dc = coords[None] - atom_coords[:, None]
    grid_dist = np.sqrt(np.einsum('ijk,ijk->ij', dc, dc))  # [natom, ngrid]

    ix, jx = np.tril_indices(natm, k=-1)

    natm, ngrid = grid_dist.shape 
    #g_ = -1 / atm_dist.reshape(natm, natm, 1) * (grid_dist.reshape(1, natm, ngrid) - grid_dist.reshape(natm, 1, ngrid))
    g_ = -1 / (atm_dist.reshape(natm, natm, 1) + np.eye(natm).reshape(natm, natm,1)) * (grid_dist.reshape(1, natm, ngrid) - grid_dist.reshape(natm, 1, ngrid))
    #g_ = jnp.array(g_)

    def pbecke_g(i, j):
      g = g_[i, j]
      charges = [elements_proton(x) for x in elements]
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

    t0 = time.time()
    #pbecke = f(gm2, gp2, natm, ngrids, ix, jx )
    pbecke = np.ones((natm, ngrids))  
    c = 0 
    # this goes up to n choose two 
    for i in range(natm): 
        for j in range(i): 
            pbecke[i] *= gm2[c]
            pbecke[j] *= gp2[c]
            c += 1
    #print("\t", time.time()-t0)
    return pbecke


def get_partition(
  mol,
  atom_coords,
  atom_grids_tab,
  radii_adjust=treutler_atomic_radii_adjust,
  atomic_radii=radi.BRAGG_RADII,
  becke_scheme=original_becke,
  concat=True, state=None
):
  t0 = time.time()
  atm_dist = inter_distance(atom_coords)  # [natom, natom]

  coords_all = []
  weights_all = []

  for ia in range(mol.natm):
    coords, vol = atom_grids_tab[mol.atom_symbol(ia)]
    coords = coords + atom_coords[ia]  # [ngrid, 3]
    pbecke  = gen_grid_partition(coords, atom_coords, mol.natm, atm_dist, mol.elements, atomic_radii)  # [natom, ngrid]
    weights = vol * pbecke[ia] / np.sum(pbecke, axis=0) 
    coords_all.append(coords)
    weights_all.append(weights)

  if concat:
    coords_all = np.vstack(coords_all)
    weights_all = np.hstack(weights_all)

  coords = (coords_all, weights_all)
  return coords_all, weights_all


class DifferentiableGrids(gen_grid.Grids):
  """Differentiable alternative to the original pyscf.gen_grid.Grids."""

  def build(self, atom_coords, state=None) :
    t0 = time.time()
    mol = self.mol

    atom_grids_tab = self.gen_atomic_grids(
      mol, self.atom_grid, self.radi_method, self.level, 
      self.prune, 
      #False, # WARNING: disables self.prune; allows changing C->O and F->N in same compute graph, but makes sizes of everythign a bit larger. 
    )

    coords, weights = get_partition(
      mol,
      atom_coords,
      atom_grids_tab,
      treutler_atomic_radii_adjust,
       self.atomic_radii,
      original_becke,
      state=state,
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


def init_dft(mol_str, opts, _coords=None, _weights=None, first=False, do_pyscf=True, state=None, pad_electrons=-1):
    do_print = False 
    #t0 = time.time()
    mol = build_mol(mol_str, opts.basis)
    if do_pyscf: pyscf_E, pyscf_hlgap, pyscf_forces = reference(mol_str, opts)
    else:        pyscf_E, pyscf_hlgap, pyscf_forces = np.zeros(1), np.zeros(1), np.zeros(1)

    N                = mol.nao_nr()                                 # N=66 for C6H6 (number of atomic **and** molecular orbitals)
    n_electrons_half = mol.nelectron//2                             # 21 for C6H6
    E_nuc            = mol.energy_nuc()                             # float = 202.4065 [Hartree] for C6H6. TODO(): Port to jax.

    from pyscf import dft
    if do_print: print("grid", end="", flush=True)

    #grids            = pyscf.dft.gen_grid.Grids(mol)
    grids            = DifferentiableGrids(mol)
    grids.level      = opts.level
    #grids.build()
    grids.build(np.concatenate([np.array(a[1]).reshape(1, 3) for a in mol._atom]), state=state)

    grid_weights    = grids.weights                                 # (grid_size,) = (45624,) for C6H6
    grid_coords     = grids.coords
    coord_str       = 'GTOval_cart_deriv1' if mol.cart else 'GTOval_sph_deriv1'
    grid_AO         = mol.eval_gto(coord_str, grids.coords, 4)      # (4, grid_size, N) = (4, 45624, 9) for C6H6.

    if do_print: print("int1e", end="", flush=True)

    kinetic         = mol.intor_symmetric('int1e_kin')              # (N,N)
    nuclear         = mol.intor_symmetric('int1e_nuc')              # (N,N)
    O               = mol.intor_symmetric('int1e_ovlp')             # (N,N)
    L               = np.linalg.cholesky(O)
    L_inv           = np.linalg.inv(L)          # (N,N)
    dm_init = pyscf.scf.hf.init_guess_by_minao(mol)

    if pad_electrons == -1: 
        init = np.eye(N)[:, :n_electrons_half] 
        mask = np.ones((1, n_electrons_half))
    else: 
        assert pad_electrons > n_electrons_half, (pad_electrons, n_electrons_half)
        init = np.eye(N)[:, :pad_electrons] 
        mask = np.zeros((1, pad_electrons))
        mask[:, :n_electrons_half] = 1

    if opts.normal: 
        ERI = mol.intor("int2e_sph")
        nonzero_distinct_ERI = np.zeros(1)
        nonzero_indices = np.zeros(1)
    else: 
        eri_threshold = 0
        batches       = 1
        nipu          = 1

        # todo: rewrite int2e_sph to only recompute changing atomic orbitals (will be N times faster). 
        if do_print: print("int2e",end ="", flush=True)
        nonzero_distinct_ERI = mol.intor("int2e_sph", aosym="s8")
        #ERI = [nonzero_distinct_ERI, nonzero_indices]
        #ERI = ERI 
        ERI = np.zeros(1)
        if do_print: print(nonzero_distinct_ERI.shape, nonzero_distinct_ERI.nbytes/10**9)
        #ERI = mol.intor("int2e_sph")
        
    def e(x): return np.expand_dims(x, axis=0)

    n_C = nao('C', opts.basis)
    n_N = nao('N', opts.basis)
    n_O = nao('O', opts.basis)
    n_F = nao('F', opts.basis)
    n_H = nao('H', opts.basis)
    n_vocab = n_C + n_N + n_O + n_F + n_H
    start, stop = 0, n_C
    c = list(range(n_vocab))[start:stop]
    start, stop = stop, stop+n_N
    n = list(range(n_vocab))[start:stop]
    start, stop = stop, stop+n_O
    o = list(range(n_vocab))[start:stop]
    start, stop = stop, stop+n_F
    f = list(range(n_vocab))[start:stop]
    start, stop = stop, stop+n_H
    h = list(range(n_vocab))[start:stop]
    types = []
    pos = []
    for a, p in mol_str:
        if a.lower() == 'h': 
            types += h
            pos += [np.array(p).reshape(1, -1)]*len(h)
        elif a.lower() == 'c': 
            types += c
            pos += [np.array(p).reshape(1, -1)]*len(c)
        elif a.lower() == 'n': 
            types += n
            pos += [np.array(p).reshape(1, -1)]*len(n)
        elif a.lower() == 'o': 
            types += o
            pos += [np.array(p).reshape(1, -1)]*len(o)
        elif a.lower() == 'f': 
            types += f
            pos += [np.array(p).reshape(1, -1)]*len(f)
        else: raise Exception()
    ao_types = np.array(types)
    pos = np.concatenate(pos)
    pad_sizes = np.zeros(1)

    state = IterationState(
        diffs_ERI = np.zeros((1,1)),
        main_grid_AO = np.zeros((1,1)),
        diffs_grid_AO = np.zeros((1,1)),
        indxs = np.zeros((1,1)),
        sparse_diffs_grid_AO = np.zeros((1,1)),
        rows = np.zeros((1,1)),
        cols = np.zeros((1,1)),
        pos=e(pos),
        ao_types=e(ao_types),
        init = e(init), 
        E_nuc=e(E_nuc), 
        ERI=e(ERI),  
        nonzero_distinct_ERI=[nonzero_distinct_ERI],
        nonzero_indices=[0],
        H_core=e(nuclear+kinetic),
        L_inv=e(L_inv), 
        L_inv_T = e(L_inv.T),
        grid_AO=e(grid_AO), 
        grid_weights=e(grid_weights), 
        grid_coords=e(grid_coords),
        pyscf_E=e(pyscf_E[-1:]), 
        N=e(mol.nao_nr()),
        mask=e(mask),
        pad_sizes=e(pad_sizes),
        precomputed_nonzero_indices=np.zeros((1,1)),
        precomputed_indxs=np.zeros((1,1)),
        forces=e(pyscf_forces),
        O = e(O),
        dm_init = e(dm_init),
    )

    return state


def summary(state): 
    if state is None: return 
    print("_"*100)
    total = 0
    for field_name, field_def in state.__dataclass_fields__.items():
        field_value = getattr(state, field_name)
        try: 
            print("%35s %24s %20s %20s"%(field_name,getattr(field_value, 'shape', None), getattr(field_value, "nbytes", None)/10**9, getattr(field_value, "dtype", None) ))
            total += getattr(field_value, "nbytes", None)/10**9

        except: 
            try: 
                print("%35s %25s %20s"%(field_name,getattr(field_value[0], 'shape', None), getattr(field_value[0], "nbytes", None)/10**9))
                total += getattr(field_value, "nbytes", None)/10**9
            except: 
                print("BROKE FOR ", field_name)
        

    print("%35s %25s %20s"%("-", "total", total))
    try:
        print(state.pyscf_E[:, -1])
    except:
        pass 
    print("_"*100)

def _cat(x,y,name):
    if "list" in str(type(x)):
        return x + y 
    else: 
        return np.concatenate([x,y])


def cat(dc1, dc2, axis=0):
    concatenated_fields = {
        field: _cat(getattr(dc1, field), getattr(dc2, field), field)
        for field in dc1.__annotations__
    }
    return IterationState(**concatenated_fields)

def _cats(xs):
    if "list" in str(type(xs[0])):
        return sum(xs, [])#x + y 
    else: 
        return np.concatenate(xs)


def cats(dcs):
    concatenated_fields = {
        field: _cats([getattr(dc, field) for dc in dcs])
        for field in dcs[0].__annotations__
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

def grad(mol, coords, weight, mo_coeff, mo_energy, dm, H, mask): # todo: will break for mask, used to be globally scoped by mistake. 
    # Initialize DFT tensors on CPU using PySCF.
    ao = pyscf.dft.numint.NumInt().eval_ao(mol, coords, deriv=2)
    eri = mol.intor("int2e_ip1")
    s1  = - mol.intor('int1e_ipovlp', comp=3)
    kin = - mol.intor('int1e_ipkin',  comp=3)
    nuc = - mol.intor('int1e_ipnuc',  comp=3)

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
    #__config__.dft_rks_RKS_grids_level = 3#opts.level
    mol = build_mol(mol_str, opts.basis)
    mol.max_cycle = 50 
    mf = pyscf.scf.RKS(mol)
    #mf.max_cycle = 50 
    #mf.xc = "b3lyp5" 
    #mf.diis_space = 8
    mf.xc = 'B3LYP5'
    mf.verbose = 0 # put this to 4 and it prints DFT options set here. 
    mf.diis_space = 8
    # options from qh9
    mf.conv_tol=1e-13
    mf.grad_tol=3.16e-5
    mf.grids.level = 3#opts.level # do we force this to -level 3?
    pyscf_energies = []
    pyscf_hlgaps = []
    lumo         = mol.nelectron//2
    homo         = lumo - 1
    t0 = time.time()
    def callback(envs):
        pyscf_energies.append(envs["e_tot"]*HARTREE_TO_EV)
        hl_gap_hartree = np.abs(envs["mo_energy"][homo] - envs["mo_energy"][lumo]) * HARTREE_TO_EV
        pyscf_hlgaps.append(hl_gap_hartree)
        print("PYSCF: ", pyscf_energies[-1], "[eV]", time.time()-t0)
    mf.callback = callback
    mf.kernel()
    print("")
    if opts.forces: 
        forces = mf.nuc_grad_method().kernel()
    else: forces = 0 
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
    mol.build(atom=mol_str, unit=unit, basis=basis_name, spin=0, verbose=0)
    return mol

def reference(mol_str, opts):
    import pickle 
    import hashlib 
    if opts.skip:  return np.zeros(1), np.zeros(1), np.zeros(1) 
    filename = "precomputed/%s.pkl"%hashlib.sha256((str(mol_str) + str(opts.basis) + str(opts.level) + unit + str(opts.forces)).encode('utf-8')).hexdigest() 
    print(filename)
    if not os.path.exists(filename):
        pyscf_E, pyscf_hlgap, pyscf_forces = pyscf_reference(mol_str, opts)
        with open(filename, "wb") as file: 
            pickle.dump([pyscf_E, pyscf_hlgap, pyscf_forces, unit], file)
    else: 
        pyscf_E, pyscf_hlgap, pyscf_forces, _ = pickle.load(open(filename, "rb"))
    return pyscf_E, pyscf_hlgap, pyscf_forces


if __name__ == "__main__":
    import os
    import argparse 

    parser = argparse.ArgumentParser()
    # DFT options 
    parser.add_argument('-basis',   type=str,   default="sto3g")  
    parser.add_argument('-level',   type=int,   default=0)

    # GD options 
    parser.add_argument('-backend', type=str,       default="cpu") 
    parser.add_argument('-lr',      type=str,     default=5e-4) 
    parser.add_argument('-min_lr',      type=str,     default=1e-7)
    parser.add_argument('-warmup_iters',      type=float,     default=1000)
    parser.add_argument('-lr_decay',      type=float,     default=200000)
    parser.add_argument('-ema',      type=float,     default=0.0)

    parser.add_argument('-steps',   type=int,       default=100000)
    parser.add_argument('-bs',      type=int,       default=8)
    parser.add_argument('-val_bs',      type=int,   default=3)
    parser.add_argument('-mol_repeats',  type=int,  default=16) # How many time to optimize wrt each molecule. 
    parser.add_argument('-grad_acc', type=int, default=0) # integer, deciding how many steps to accumulate. 
    parser.add_argument('-shuffle',  action="store_true") # whether to to shuffle the window of states each step. 

    # energy computation speedups 
    parser.add_argument('-foriloop',  action="store_true") # whether to use jax.lax.foriloop for sparse_symmetric_eri (faster compile time but slower training. )
    parser.add_argument('-xc_f32',   action="store_true") 
    parser.add_argument('-eri_f32',  action="store_true") 
    parser.add_argument('-nn_f32',  action="store_true") 
    parser.add_argument('-eri_bs',  type=int, default=8) 

    parser.add_argument('-loss_vxc',  action="store_true")

    parser.add_argument('-normal',     action="store_true") 
    parser.add_argument('-wandb',      action="store_true") 
    parser.add_argument('-prof',       action="store_true") 
    parser.add_argument('-visualize',  action="store_true") 
    parser.add_argument('-skip',       action="store_true", help="skip pyscf test case") 

    # dataset 
    parser.add_argument('-nperturb',  type=int, default=0, help="How many atoms to perturb (supports 1,2,3)") 
    parser.add_argument('-qm9',        action="store_true") 
    parser.add_argument('-md17',       type=int, default=-1) 
    parser.add_argument('-qh9',        action="store_true") 
    parser.add_argument('-benzene',        action="store_true") 
    parser.add_argument('-hydrogens',        action="store_true") 
    parser.add_argument('-water',        action="store_true") 
    parser.add_argument('-waters',        action="store_true") 
    parser.add_argument('-alanine',        action="store_true") 
    parser.add_argument('-do_print',        action="store_true")  # useful for debugging. 
    parser.add_argument('-states',         type=int,   default=1)
    parser.add_argument('-workers',        type=int,   default=5) 
    parser.add_argument('-precompute',        action="store_true")  # precompute labels; only run once for data{set/augmentation}.
    parser.add_argument('-wiggle_var',     type=float,   default=0.05, help="wiggle N(0, wiggle_var), bondlength=1.5/30")
    parser.add_argument('-eri_threshold',  type=float,   default=1e-10, help="loss function threshold only")
    parser.add_argument('-rotate_deg',     type=float,   default=90, help="how many degrees to rotate")
    parser.add_argument('-test_dataloader',     action="store_true", help="no training, just test/loop through dataloader. ")

    # md 
    parser.add_argument('-md_T',  type=int,   default=300, help="temperature for md in Kelvin [K].")
    parser.add_argument('-md_time',  type=float,   default=0.002, help="time step for md in picoseconds [ps].")

    # models 
    parser.add_argument('-nn',       action="store_true", help="train nn, defaults to GD") 
    parser.add_argument('-tiny',     action="store_true") 
    parser.add_argument('-small',    action="store_true") 
    parser.add_argument('-base',     action="store_true") 
    parser.add_argument('-medium',   action="store_true") 
    parser.add_argument('-large',    action="store_true") 
    parser.add_argument('-xlarge',   action="store_true") 
    parser.add_argument('-largep',   action="store_true")  # large "plus"
    parser.add_argument('-forces',   action="store_true")  
    parser.add_argument("-checkpoint", default=-1, type=int, help="which iteration to save model (default -1 = no saving)") # checkpoint model 
    parser.add_argument("-resume",   default="", help="path to checkpoint pickle file") # resume saved (checkpointed) model
    parser.add_argument("-inference",   default=0, type=int)
    opts = parser.parse_args()
    # trick to allow 1e-4/math.sqrt(16) when reducing bs by 16. 
    opts.lr = eval(opts.lr)
    opts.min_lr = eval(opts.min_lr)
    if opts.tiny or opts.small or opts.base or opts.large or opts.xlarge: opts.nn = True 

    assert opts.grad_acc == 0 or opts.mol_repeats % opts.grad_acc == 0, "mol_repeats needs to be a multiple of grad_acc (gradient accumulation)."

    class HashableNamespace:
      def __init__(self, namespace): self.__dict__.update(namespace.__dict__)
      def __hash__(self): return hash(tuple(sorted(self.__dict__.items())))
    opts = HashableNamespace(opts)

    args_dict = vars(opts)
    print(args_dict)

    if opts.qm9: 
        df = pd.read_pickle("alchemy/atom_9.pickle")
        df = df[df["spin"] == 0] # only consider spin=0
        mol_strs = df["pyscf"].values

    if opts.qh9: 
        mol_strs = []

    # benzene 
    if opts.benzene: 
        mol_strs = [[
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
            ]]
    # hydrogens 
    if opts.hydrogens: 
        mol_strs = [[
                ["H", ( 0.0000,  0.0000, 0.0000)],
                ["H", ( 1.4000,  0.0000, 0.0000)],
            ]]
    if opts.md17 > 0 : 
        mol_strs = [[
                ["O", ( 0.0000,  0.0000, 0.0000)],
                ["H", ( 0.0000,  1.4000, 0.0000)],
                ["H", ( 1.4000,  0.0000, 0.0000)],
            ]]
    if opts.waters: 
        mol_strs = [[
            ["O",    (-1.464,  0.099,  0.300)],
            ["H",    (-1.956,  0.624, -0.340)],
            ["H",    (-1.797, -0.799,  0.206)],
            ["O",    ( 1.369,  0.146, -0.395)],
            ["H",    ( 1.894,  0.486,  0.335)],
            ["H",    ( 0.451,  0.165, -0.083)]]]

    elif opts.alanine: 
        mol_strs = [[ # 22 atoms (12 hydrogens) => 10 heavy atoms (i.e. larger than QM9). 
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


    nanoDFT_E, (nanoDFT_hlgap, mo_energy, mo_coeff, grid_coords, grid_weights, dm, H) = nanoDFT(mol_strs, opts)

    exit()
    pyscf_E, pyscf_hlgap, pyscf_forces = reference(mol_str, opts)
    nanoDFT_forces = grad(mol, grid_coords, grid_weights, mo_coeff, mo_energy, np.array(dm), np.array(H))
    print_difference(nanoDFT_E, nanoDFT_forces, 0 , nanoDFT_hlgap, pyscf_E, pyscf_forces, pyscf_hlgap)