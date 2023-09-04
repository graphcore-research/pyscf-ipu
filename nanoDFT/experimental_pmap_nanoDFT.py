# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import os 
os.environ["OMP_NUM_THREADS"] = "16" 
import jax
import jax.numpy as jnp
import numpy as np
import pyscf
from jsonargparse import CLI, Namespace
from exchange_correlation.b3lyp import b3lyp
from electron_repulsion.direct import prepare_electron_repulsion_integrals, electron_repulsion_integrals, ipu_einsum
from functools import partial

HARTREE_TO_EV = 27.2114079527
EPSILON_B3LYP = 1e-20
HYB_B3LYP = 0.2

def energy(density_matrix, H_core, diff_JK, E_xc, E_nuc, _np=jax.numpy):
    """Density Functional Theory (DFT) solves the optimisation problem:

        min_{density_matrix} energy(density_matrix, ...)

    We like to think of `energy(...)` as a loss function. `density_matrix`
    represents the density of electrons as:

        rho(r) = sum_{ij}^N density_matrix_{ij} X_i(r) X_j(r)  where X_i(r)~exp(-|r|^2).

    Here N is the number of atomic orbitals (AO) **and** molecular orbitals (N=66 for C6H6).
    All input matrices (density_matrix, H_core, diff_JK) are (N, N). The X_i(r) are called
    Gaussian Type Orbitals (GTO). The inputs (diff_JK, E_xc) depend on density_matrix.
    """
    E_core = _np.sum(density_matrix * H_core)             # float = -712.04[Ha] for C6H6.
    E_J_K    = _np.sum(density_matrix * diff_JK)          # NOTE: diff_JK is already diff_JK = J - (K / 2 * HYB_B3LYP)
    E      = E_core + E_J_K/2 + E_xc + E_nuc              # float = -232.04[Ha] for C6H6.

    return _np.array([E, E_core, E_J_K/2, E_xc, E_nuc])   # Energy (and its terms).

def nanoDFT_iteration(i, vals, opts, mol):
    """Each call updates density_matrix attempting to minimize energy(density_matrix, ... ). """
    density_matrix, V_xc, O, H_core, L_inv, diff_JK               = vals[:6]                                                          # All (N, N) matrices
    E_nuc, occupancy, ERI, sharded_grid_weights, sharded_grid_AO, diis_history, log= vals[6:]                  # Varying types/shapes.

    if opts.v: 
        print("---------- MEMORY CONSUMPTION ----------")
        MB = 0
        for t in vals:
            try: 
                if type(t) != type(()) and len(np.shape(t)) > 0: 
                    print(t.nbytes/10**6, t.shape)
                    MB += t.nbytes/10**6
            except: 
                print(type(t)) 
        print("ERI")
        for a in ERI: # prints weird in dense_ERI case 
            print( a.nbytes/10**6, a.shape)
            MB += a.nbytes / 10**6
        print("__________")
        print("Total: ", MB)  
        print("----------------------------------------")
        print("")

    # Step 1: Update Hamiltonian (optionally use DIIS to improve DFT convergence).
    H = H_core + diff_JK + V_xc                                                                 # (N, N)
    if opts.diis: H, diis_history = DIIS(i, H, density_matrix, O, diis_history, opts)           # H_{i+1}=c_1H_i+...+c9H_{i-9}.

    # Step 2: Solve eigh (L_inv turns generalized eigh into eigh).
    eigvects = L_inv.T @ linalg_eigh(L_inv @ H @ L_inv.T, opts)[1]                              # (N, N)

    # Step 3: Use result from eigenproblem to update density_matrix.
    density_matrix = (eigvects*occupancy*2) @ eigvects.T                                        # (N, N)
    E_xc, V_xc     = exchange_correlation(density_matrix, sharded_grid_AO, sharded_grid_weights)                # float (N, N)
    diff_JK        = get_JK(density_matrix, ERI, opts.dense_ERI)    #(4,gsize,N)                                 # (N, N)

    # Log SCF matrices and energies (not used by DFT algorithm).
    #log["matrices"] = log["matrices"].at[i].set(jnp.stack((density_matrix, J, K, H)))           # (iterations, 4, N, N) ## removing this reduces memory from 50kb to 18kb
    N = density_matrix.shape[0]
    log["matrices"] = jax.lax.dynamic_update_slice(log["matrices"], density_matrix.reshape(1, 1, N, N), (i, 0, 0, 0))
    log["matrices"] = jax.lax.dynamic_update_slice(log["matrices"], diff_JK.       reshape(1, 1, N, N), (i, 1, 0, 0))
    log["matrices"] = jax.lax.dynamic_update_slice(log["matrices"], H.             reshape(1, 1, N, N), (i, 2, 0, 0))
    log["energy"]   = log["energy"].at[i].set(energy(density_matrix, H_core, diff_JK, E_xc, E_nuc))  # (iterations, 6)

    return [density_matrix, V_xc, O, H_core, L_inv, diff_JK, E_nuc, occupancy, ERI, sharded_grid_weights, sharded_grid_AO, diis_history, log]

def exchange_correlation(density_matrix, sharded_grid_AO, sharded_grid_weights):
    """Compute exchange correlation integral using atomic orbitals (AO) evalauted on a grid. """
    # TODO: we can also use sparsity on grid! Visualization shows it's similarly sparse to ERI! 

    # Perfectly SIMD parallelizable over grid_size axis.
    # Only need one reduce_sum in the end. 
    grid_AO_dm = sharded_grid_AO[0] @ density_matrix # (gsize, N) @ (N, N) -> (gsize, N)
    grid_AO_dm = jnp.expand_dims(grid_AO_dm, axis=0) # (1, gsize, N)
    mult = grid_AO_dm * sharded_grid_AO  
    rho = jnp.sum(mult, axis=2)                                # (4, grid_size)=(4, 45624) for C6H6.
    E_xc, vrho, vgamma = b3lyp(rho, EPSILON_B3LYP)                                               # (gridsize,) (gridsize,) (gridsize,)
    E_xc = jax.lax.psum(jnp.sum(rho[0] * sharded_grid_weights * E_xc), axis_name="p")                                                 # float=-27.968[Ha] for C6H6 at convergence.
    rho = jnp.concatenate([vrho.reshape(1, -1)/2, 4*vgamma*rho[1:4]], axis=0) * sharded_grid_weights     # (4, grid_size)=(4, 45624)
    grid_AO_T = sharded_grid_AO[0].T # (N, gsize)
    rho = jnp.expand_dims(rho, axis=2) # (4, gsize, 1)
    grid_AO_rho = sharded_grid_AO * rho # (4, gsize, N)
    sum_grid_AO_rho = jnp.sum(grid_AO_rho, axis=0) # (gsize, N)
    V_xc = grid_AO_T @ sum_grid_AO_rho # (N, N)
    V_xc = jax.lax.psum(V_xc, axis_name="p") #(N, N)
    V_xc = V_xc + V_xc.T                                                                         # (N, N)
    return E_xc, V_xc                                                                            # (float) (N, N)

def get_JK(density_matrix, ERI, dense_ERI):
    """Computes the (N, N) matrices J and K. Density matrix is (N, N) and ERI is (N, N, N, N).  """
    N = density_matrix.shape[0]

    if dense_ERI: 
        J = jnp.einsum('ijkl,ji->kl', ERI, density_matrix)                                       # (N, N)
        K = jnp.einsum('ijkl,jk->il', ERI, density_matrix)                                       # (N, N)
        diff_JK = J - (K / 2 * HYB_B3LYP)
    else:
        from sparse_symmetric_ERI import sparse_symmetric_einsum
        diff_JK = sparse_symmetric_einsum(ERI[0], ERI[1], density_matrix, opts.backend)
        
    diff_JK = diff_JK.reshape(N, N)

    return diff_JK                                                                      # (N, N) (N, N),

def _nanoDFT(E_nuc, density_matrix, kinetic, nuclear, O, 
              mask, _input_floats, _input_ints, L_inv, diis_history, sharded_grid_AO, sharded_grid_weights, ERI, opts, mol):
    # Utilize the IPUs MIMD parallism to compute the electron repulsion integrals (ERIs) in parallel.
    #if opts.backend == "ipu": ERI = electron_repulsion_integrals(_input_floats, _input_ints, mol, opts.threads_int, opts.intv)
    #else: pass # Compute on CPU.
    sharded_grid_AO = jnp.transpose(sharded_grid_AO, (1,0,2)) # (padded_gsize/16, 4, N) -> (4, pgsize, N)
        
    # Precompute the remaining tensors.
    E_xc, V_xc = exchange_correlation(density_matrix, sharded_grid_AO, sharded_grid_weights) # float (N, N)
    diff_JK  = get_JK(density_matrix, ERI, opts.dense_ERI)                      # (N, N) (N, N)
    H_core     = kinetic + nuclear                                           # (N, N)

    # Log matrices from all DFT iterations (not used by DFT algorithm).
    N = H_core.shape[0]
    log = {"matrices": np.zeros((opts.its, 3, N, N)), "E_xc": np.zeros((opts.its)), "energy": np.zeros((opts.its, 5))}

    # Perform DFT iterations.
    log = jax.lax.fori_loop(0, opts.its, partial(nanoDFT_iteration, opts=opts, mol=mol), [density_matrix, V_xc, O, H_core, L_inv, diff_JK, # all (N, N) matrices
                                                        E_nuc, mask, ERI, sharded_grid_weights, sharded_grid_AO, diis_history, log])[-1]
    return log["matrices"], H_core, log["energy"]

def init_dft_tensors_cpu(mol, opts, DIIS_iters=9):
    N                = mol.nao_nr()                                 # N=66 for C6H6 (number of atomic **and** molecular orbitals)
    print("-----> [ %i ] <-----"%N)
    n_electrons_half = mol.nelectron//2                             # 21 for C6H6
    E_nuc            = mol.energy_nuc()                             # float = 202.4065 [Hartree] for C6H6. 

    # TODO(): port grid/eval_gto to Jax.
    grids            = pyscf.dft.gen_grid.Grids(mol)
    grids.level      = opts.level
    grids.build()
    grid_weights    = grids.weights                                 # (grid_size,) = (45624,) for C6H6
    coord_str       = 'GTOval_cart_deriv1' if mol.cart else 'GTOval_sph_deriv1'
    grid_AO         = mol.eval_gto(coord_str, grids.coords, 4)      # (4, grid_size, N) = (4, 45624, 9) for C6H6.
    density_matrix  = pyscf.scf.hf.init_guess_by_minao(mol)         # (N,N)=(66,66) for C6H6.

    # TODO(): Add integral math formulas for kinetic/nuclear/O/ERI.
    kinetic         = mol.intor_symmetric('int1e_kin')              # (N,N)
    nuclear         = mol.intor_symmetric('int1e_nuc')              # (N,N)
    O               = mol.intor_symmetric('int1e_ovlp')             # (N,N)
    L_inv           = np.linalg.inv(np.linalg.cholesky(O))          # (N,N)

    input_floats, input_ints = 0, 0 #prepare_electron_repulsion_integrals(mol)[:2]
    mask = np.concatenate([np.ones(n_electrons_half), np.zeros(N-n_electrons_half)])

    # DIIS is an optional technique to improve DFT convergence.
    DIIS_H       = np.zeros((DIIS_iters+1, DIIS_iters+1))
    DIIS_H[0,1:] = DIIS_H[1:,0] = 1
    diis_history = (np.zeros((DIIS_iters, N**2)), np.zeros((DIIS_iters, N**2)), DIIS_H)

    tensors = [E_nuc, density_matrix, kinetic, nuclear, O, mask, input_floats, input_ints, L_inv, diis_history]

    return tensors, n_electrons_half, E_nuc, N, L_inv, grid_weights, grids.coords, grid_AO

def nanoDFT(mol, opts):
    # Init DFT tensors on CPU using PySCF.
    tensors, n_electrons_half, E_nuc, N, L_inv, grid_weights, grid_coords, grid_AO = init_dft_tensors_cpu(mol, opts)

    sharded_grid_AO = jnp.transpose(grid_AO, (1, 0, 2)) # (4,gsize,N) -> (gsize,4,N)
    sharded_grid_weights = grid_weights
    gsize = sharded_grid_AO.shape[0]

    remainder = gsize % opts.ndevices
    if remainder != 0: 
        sharded_grid_AO = jnp.pad(sharded_grid_AO, ((0,remainder), (0,0), (0,0)) )
        sharded_grid_weights = jnp.pad(sharded_grid_weights, ((0,remainder)) )
    sharded_grid_AO = sharded_grid_AO.reshape(opts.ndevices, -1, 4, N)
    sharded_grid_weights = sharded_grid_weights.reshape(opts.ndevices, -1)

    tensors.append(sharded_grid_AO)
    tensors.append(sharded_grid_weights)

    # Run DFT algorithm (can be hardware accelerated).
    # we can have "ipus" argument instead, or pod16 as backend? 
    if opts.dense_ERI: 
        assert opts.ndevices == 1, "Only support '--dense_ERI True' for `--ndevices 1`. "
        eri_in_axes = 0
        ERI = mol.intor("int2e_sph")
        ERI = np.expand_dims(ERI, 0)
        tensors.append(ERI)
    else: 
        from sparse_symmetric_ERI import vmap_num_repetitions_fast
        distinct_ERI         = mol.intor("int2e_sph", aosym="s8")
        print(distinct_ERI.size)
        indxs = np.abs(distinct_ERI)<1e-7
        distinct_ERI[indxs] = 0
        nonzero_indices      = np.nonzero(distinct_ERI)[0].astype(np.int32)
        nonzero_distinct_ERI = distinct_ERI[nonzero_indices].astype(np.float32)
        rep = jax.jit(vmap_num_repetitions_fast, backend="cpu")(nonzero_indices)
        nonzero_distinct_ERI = nonzero_distinct_ERI / rep
        remainder = nonzero_indices.shape[0] % (opts.ndevices*opts.batches)
        if remainder != 0:
            nonzero_indices = np.pad(nonzero_indices, (0,opts.ndevices*opts.batches-remainder))
            nonzero_distinct_ERI = np.pad(nonzero_distinct_ERI, (0,opts.ndevices*opts.batches-remainder))
        nonzero_indices = nonzero_indices.reshape(opts.ndevices, opts.batches, -1)
        nonzero_distinct_ERI = nonzero_distinct_ERI.reshape(opts.ndevices, opts.batches, -1)
        sparse_ERI = [nonzero_distinct_ERI, nonzero_indices]
        tensors.append(sparse_ERI)
        eri_in_axes = [0,0]

    jitted_nanoDFT = jax.pmap(partial(_nanoDFT, opts=opts, mol=mol), axis_name="p", backend=opts.backend,
                            in_axes=(None, None, None, None, None, None, None, None, None, (None, None, None), 0, 0, eri_in_axes))

    jitted_nanoDFT(*tensors)
    vals = jitted_nanoDFT(*tensors)
    logged_matrices, H_core, logged_energies = [np.asarray(a[0]).astype(np.float64) for a in vals] # Ensure CPU

    # It's cheap to compute energy/hlgap on CPU in float64 from the logged values/matrices.
    logged_E_xc = logged_energies[:, 3].copy()
    print(logged_energies[:, 0] * HARTREE_TO_EV)

    density_matrices, diff_JKs, H = [logged_matrices[:, i] for i in range(3)]
    energies, hlgaps = np.zeros((opts.its, 5)), np.zeros(opts.its)
    for i in range(opts.its):
        energies[i] = energy(density_matrices[i], H_core, diff_JKs[i], logged_E_xc[i], E_nuc, np)
        hlgaps[i]   = hlgap(L_inv, H[i], n_electrons_half, np)
    energies, logged_energies, hlgaps = [a * HARTREE_TO_EV for a in [energies, logged_energies, hlgaps]]
    mo_energy, mo_coeff = np.linalg.eigh(L_inv @ H[-1] @ L_inv.T)
    mo_coeff = L_inv.T @ mo_coeff
    return energies, (logged_energies, hlgaps, mo_energy, mo_coeff, grid_coords, grid_weights)

def DIIS(i, H, density_matrix, O, diis_history, opts):
    # DIIS is an optional technique which improves DFT convergence by computing:
    #   H_{i+1} = c_1 H_i + ... + c_8 H_{i-8}  where  c=pinv(some_matrix)[0,:]
    # We thus like to think of DIIS as "fancy momentum".
    _V, _H, DIIS_H     = diis_history       # (diis_iters, N**2), (diis_iters, N**2), (diis_iters+1, diis_iters+1)
    diis_iters, d = _V.shape
    DIIS_head = i % _V.shape[0]             # int in {0, ..., diis_iters-1}
    sdf       = O @ density_matrix @ H      # (N, N)=(66,66) for C6H6.
    errvec    = sdf - sdf.T                 # (N, N)

    _V = jax.lax.dynamic_update_slice(_V, errvec.reshape(1, d), (DIIS_head, 0))     # (diis_iters, N**2)=(9, 4356) for C6H6.
    _H = jax.lax.dynamic_update_slice(_H, H.reshape(1, d),      (DIIS_head, 0))     # (diis_iters, N**2)

    mask = jnp.where(np.arange(_V.shape[0]) < jnp.minimum(i+1, _V.shape[0]), jnp.ones(_V.shape[0], dtype=_V.dtype), jnp.zeros(_V.shape[0], dtype=_V.dtype))
    tmps = (_V.reshape(diis_iters, 1, d) @ errvec.reshape(1, d, 1)).reshape(-1) * mask # (diis_iters, )

    DIIS_H = jax.lax.dynamic_update_slice( DIIS_H, tmps.reshape(1, -1), (DIIS_head+1, 1) ) # (diis_iters+1, diis_iters+1)
    DIIS_H = jax.lax.dynamic_update_slice( DIIS_H, tmps.reshape(-1, 1), (1, DIIS_head+1) ) # (diis_iters+1, diis_iters+1)

    mask_         = jnp.concatenate([jnp.ones(1, dtype=mask.dtype), mask]) # (diis_iters+1,)
    masked_DIIS_H = DIIS_H * mask_.reshape(-1, 1) * mask_.reshape(1, -1)

    if opts.backend == "ipu":  c = pinv0(masked_DIIS_H, opts)                 # (diis_iters+1,)=10 for C6H6.
    else:                      c = jnp.linalg.pinv(masked_DIIS_H)[0, :]       # (diis_iters+1,)=10 for C6H6.

    H = (c[1:] @ _H).reshape(H.shape)                                         # (N, N)
    return H, (_V, _H, DIIS_H)                                                # (N, N)

def hlgap(L_inv, H, n_electrons_half, _np):
    mo_energy   = _np.linalg.eigh(L_inv @ H @ L_inv.T)[0]
    return _np.abs(mo_energy[n_electrons_half] - mo_energy[n_electrons_half-1])

def linalg_eigh(x, opts):
    if opts.backend == "ipu":
        from tessellate_ipu.linalg import ipu_eigh

        n = x.shape[0]
        pad = n % 2
        if pad:
            x = jnp.pad(x, [(0, 1), (0, 1)], mode='constant')
            assert False 

        eigvects, eigvals = ipu_eigh(x, sort_eigenvalues=True, num_iters=12) # is this the culprit? 

        if pad:
            e1 = eigvects[-1:]
            col = jnp.argmax(e1)
            eigvects = jnp.roll(eigvects, -col-1)
            eigvects = eigvects[:, :-1]
            eigvects = jnp.roll(eigvects, -(-col))
            eigvects = eigvects[:-1]
            assert False 
    else:
        eigvals, eigvects = jnp.linalg.eigh(x)

    return eigvals, eigvects

def pinv0(a, opts):  # take out first row
    # TODO: add a comment explaining the role of this constant
    cond =  9*1.1920929e-07
    vals, vect = linalg_eigh(a, opts)
    c = vect @ ( jnp.where( jnp.abs(vals) > cond, 1/vals, 0) * vect[0, :])
    return c

def grad_elec(weight, grid_AO, eri, s1, h1aos, natm, aoslices, mask, mo_energy, mo_coeff):
    # Electronic part of RHF/RKS gradients
    dm0  = 2 * (mo_coeff*mask) @ mo_coeff.T                                 # (N, N) = (66, 66) for C6H6.
    dme0 = 2 * (mo_coeff * mask*mo_energy) @  mo_coeff.T                    # (N, N) = (66, 66) for C6H6>

    # Code identical to exchange correlation.
    rho             = jnp.sum( grid_AO[:1] @ dm0 * grid_AO, axis=2)         # (10, grid_size) = (10, 45624) for C6H6.
    _, vrho, vgamma = b3lyp(rho, EPSILON_B3LYP)                             # (grid_size,) (grid_size,)
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

def grad(mol, coords, weight, mo_coeff, mo_energy):
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

    _grad_elec = jax.jit(grad_elec, static_argnames=["aoslices", "natm"], backend="cpu")
    _grad_nuc = jax.jit(grad_nuc, backend="cpu")

    return _grad_elec(weight, ao, eri, s1, h1aos, mol.natm, tuple([tuple(a) for a in aoslices.tolist()]), mask, mo_energy, mo_coeff)  + _grad_nuc(charges, coords)

def pyscf_reference(mol_str, opts):
    from pyscf import __config__
    __config__.dft_rks_RKS_grids_level = opts.level

    mol = build_mol(mol_str, opts.basis)
    mol.max_cycle = opts.its
    mf = pyscf.scf.RKS(mol)
    mf.max_cycle = opts.its
    mf.xc = opts.xc
    mf.diis_space = 9
    if not opts.diis:  #
        mf.diis_space = 0
        mf.diis = False
    pyscf_energies = []
    pyscf_hlgaps = []
    lumo         = mol.nelectron//2
    homo         = lumo - 1
    def callback(envs):
        pyscf_energies.append(envs["e_tot"]*HARTREE_TO_EV)
        hl_gap_hartree = np.abs(envs["mo_energy"][homo] - envs["mo_energy"][lumo]) * HARTREE_TO_EV
        pyscf_hlgaps.append(hl_gap_hartree)
    mf.callback = callback
    mf.kernel()
    forces = 0#mf.nuc_grad_method().kernel()
    return np.array(pyscf_energies), np.array(pyscf_hlgaps), np.array(forces)

def print_difference(nanoDFT_E, nanoDFT_forces, nanoDFT_logged_E, nanoDFT_hlgap, pyscf_E, pyscf_forces, pyscf_hlgap):
    #TODO(HH): rename to match caller variable names
    print("pyscf_hlgap\t%15f"%( pyscf_hlgap[-1]))
    print("us_hlgap\t%15f"%(    nanoDFT_hlgap[-1]))
    print("err_hlgap\t%15f"%np.abs(pyscf_hlgap[-1]  - nanoDFT_hlgap[-1]))
    print("pyscf:\t\t%15f"%pyscf_E[-1])
    print("us:\t\t%15f"%nanoDFT_E[-1, 0])
    print("mus:\t\t%15f"%np.mean(nanoDFT_E[-10:, 0]))
    print("diff:\t\t%15f"%np.abs(pyscf_E[-1]-nanoDFT_E[-1, 0]))
    print("mdiff:\t\t%15f"%np.abs(pyscf_E[-1]-np.mean(nanoDFT_E[-10:, 0])), np.std(nanoDFT_E[-10:, 0]))
    print("chemAcc: \t%15f"%0.043)
    print("chemAcc/diff: \t%15f"%(0.043/np.abs(pyscf_E[-1]-nanoDFT_E[-1, 0])))
    print("chemAcc/mdiff: \t%15f"%(0.043/np.abs(pyscf_E[-1]-np.mean(nanoDFT_E[-10:, 0]))))
    print("")
    pyscf_E = np.concatenate([pyscf_E, np.ones(nanoDFT_E.shape[0]-pyscf_E.shape[0])*pyscf_E[-1]])
    pyscf_hlgap = np.concatenate([pyscf_hlgap, np.ones(nanoDFT_hlgap.shape[0]-pyscf_hlgap.shape[0])*pyscf_hlgap[-1]])
    print("%18s"%"", "\t".join(["%10s"%str("iter %i "%i) for i in np.arange(1, nanoDFT_E.shape[0]+1)[1::3]]))
    print("%18s"%"Error Energy [eV]", "\t".join(["%10s"%str("%.2e"%f) for f in (pyscf_E[1::3] - nanoDFT_E[1::3, 0]).reshape(-1)]))
    print("%18s"%"Error HLGAP [eV]", "\t".join(["%10s"%str("%.2e"%f) for f in (pyscf_hlgap[1::3]   - nanoDFT_hlgap[1::3]).reshape(-1)]))

    print()
    print("%18s"%"E_core [eV]", "\t".join(["%10s"%str("%.5f"%f) for f in (nanoDFT_E[1::3, 1]).reshape(-1)]))
    print("%18s"%"E_J_K [eV]", "\t".join(["%10s"%str("%.5f"%f) for f in (nanoDFT_E[1::3, 2]).reshape(-1)]))
    print("%18s"%"E_xc [eV]", "\t".join(["%10s"%str("%.5f"%f) for f in (nanoDFT_E[1::3, 3]).reshape(-1)]))
    print("%18s"%"E_nuc [eV]", "\t".join(["%10s"%str("%.5f"%f) for f in (nanoDFT_E[1::3, 4]).reshape(-1)]))


def build_mol(mol_str, basis_name):
    mol = pyscf.gto.mole.Mole()
    mol.build(atom=mol_str, unit="Angstrom", basis=basis_name, spin=0, verbose=0)
    return mol

def nanoDFT_options(
        its: int = 20,
        mol_str: str = "benzene",
        float32: bool = False,
        basis: str = "sto-3g",
        xc: str = "b3lyp",
        backend: str = "cpu",
        level: int = 0,
        multv: int = 2,
        intv: int = 1,
        threads: int = 1,
        threads_int: int = 1,
        diis: bool = True,
        structure_optimization: bool = False, # AKA gradient descent on energy wrt nuclei
        batches: int = 32,
        ndevices: int = 1, 
        dense_ERI: bool = False, # whether to use sparse/distinct eri. 
        v: bool = False, # verbose 
        profile: bool = False # if we only want profile exit after IPU finishes. 
        
):
    """
    nanoDFT

    Args:
        its (int): Number of Kohn-Sham iterations.
        mol_str (str): Molecule string, e.g., "H 0 0 0; H 0 0 1; O 1 0 0; "
        float32 (bool) : Whether to use float32 (default is float64).
        basis (str): Which Gaussian basis set to use.
        xc (str): Exchange-correlation functional. Only support B3LYP
        backend (str): Accelerator backend to use: "-backend cpu" or "-backend ipu".
        level (int): Level of grids for XC numerical integration.
        gdb (int): Which version of GDP to load {10, 11, 13, 17}.
        multv (int): Which version of our einsum algorithm to use;comptues ERI@flat(v). Different versions trades-off for memory vs sequentiality
        intv (int): Which version to use of our integral algorithm.
        threads (int): For -backend ipu. Number of threads for einsum(ERI, dm) with custom C++ (trades-off speed vs memory).
        threads_int (int): For -backend ipu. Number of threads for computing ERI with custom C++ (trades off speed vs memory).
    """
    if mol_str == "benzene":
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
    elif mol_str == "methane":
        mol_str = [
            ["C", (0, 0, 0)],
            ["H", (0, 0, 1)],
            ["H", (0, 1, 0)],
            ["H", (1, 0, 0)],
            ["H", (1, 1, 1)]
        ]
    elif mol_str == "pmap":
        mol_str = "".join([f"H 0 0 {i};" for i in range(2)]) # N=4 
        # mol_str = [["H", (0, 0, n) for n in range(2)]]          # N=4
        # mol_str = [["H", (0, 0, n) for n in range(8)]]          # N=16
        # mol_str = [["H", (0, 0, n) for n in range(8*2)]]        # N=32
        # mol_str = [["H", (0, 0, n) for n in range(32)]]         # N=64
        # mol_str = [["H", (0, 0, n) for n in range(32+16)]]      # N=96
        # mol_str = [["H", (0, 0, n) for n in range(8*2*2*2)]]    # N=128
        # mol_str = [["H", (0, 0, n) for n in range(8*2*2*2*2)]]  # N=256
        basis = "6-31g"
    elif mol_str == "cgrid": 
        mol_str = [["C", (0, i*1.5, j*1.5)] for i in range(10) for j in range(10)]
        #mol_str = [["C", (0, i*1.5, j*1.5)] for i in range(2) for j in range(2)]
        #mol_str = [["C", (0, i*1.5, j*1.5)] for i in range(5) for j in range(6)]
        #mol_str = [["C", (0, i*1.5, j*1.5)] for i in range(5) for j in range(5)]
        #mol_str = [["C", (0, i*1.5, j*1.5)] for i in range(6) for j in range(6)]
        #mol_str = [["C", (0, i*1.5, j*1.5)] for i in range(7) for j in range(7)] 
    
    args = locals()
    mol_str = args["mol_str"]
    del args["mol_str"]
    args = Namespace(**args)
    if not args.float32:
        jax.config.update('jax_enable_x64', not float32)
    return args, mol_str

if __name__ == "__main__":
    # Limit PySCF threads to mitigate problem with NUMA nodes.
    jax.config.FLAGS.jax_platform_name = 'cpu'
    opts, mol_str = CLI(nanoDFT_options)
    assert opts.xc == "b3lyp"
    print("float32") if opts.float32 else print("float64")

    if not opts.structure_optimization:
        # Test Case: Compare nanoDFT against PySCF.
        mol = build_mol(mol_str, opts.basis)
        nanoDFT_E, (nanoDFT_logged_E, nanoDFT_hlgap, mo_energy, mo_coeff, grid_coords, grid_weights) = nanoDFT(mol, opts)
        nanoDFT_forces = 0 # grad(mol, grid_coords, grid_weights, mo_coeff, mo_energy)
        pyscf_E, pyscf_hlgap, pyscf_forces = pyscf_reference(mol_str, opts)
        print_difference(nanoDFT_E, nanoDFT_forces, nanoDFT_logged_E, nanoDFT_hlgap, pyscf_E, pyscf_forces, pyscf_hlgap)
    else:
        # pip install mogli imageio[ffmpeg] matplotlib
        import mogli
        import imageio
        import matplotlib.pyplot as plt
        opts.basis = "6-31G"
        p = np.array([[0,1,1], [0,2,2], [0,3,3],
                      [0,4,4], [0,5,5], [0,6,6]])
        np.random.seed(42)
        p = p + np.random.normal(0, 0.3, p.shape) # slightly break symmetry
        A = ["H", "O", "H", "H", "O", "H"]
        natm = p.shape[0]
        os.makedirs("_tmp/", exist_ok=True)
        E = []
        ims = []
        for i in range(20):
            mol_str = "".join([f"{A[i]} {p[i]};".replace("[", "]").replace("]", "") for i in range(natm)])
            mol = build_mol(mol_str, opts.basis)
            nanoDFT_E, (nanoDFT_logged_E, nanoDFT_hlgap, mo_energy, mo_coeff, grid_coords, grid_weights) = nanoDFT(mol, opts)
            f = open(f"_tmp/{i}.xyz", "w")
            f.write(f"""{natm}\n{mol_str} {nanoDFT_E[-1, 0]}\n"""+"".join([f"{A[i]} {p[i]}\n".replace("[", "").replace("]", "") for i in range(natm)]))
            f.close()
            molecules = mogli.read(f'_tmp/{i}.xyz')
            mogli.export(molecules[0], f'_tmp/{i}.png', width=400, height=400,
                 bonds_param=1.15, camera=((9, 0, 0),
                                           (0, 0, 0),
                                           (0, 9, 0)))
            ims.append(imageio.v2.imread(f"_tmp/{i}.png/"))
            E.append(nanoDFT_E[-1, 0])
            nanoDFT_forces = grad(mol, grid_coords, grid_weights, mo_coeff, mo_energy)
            p = p - nanoDFT_forces
            print(nanoDFT_E[-1, 0], i)
        writer = imageio.get_writer('_tmp/test.gif', loop=0, duration=3)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for c, i in enumerate(ims):
            for a in ax: a.cla()
            ax[0].axis("off")
            ax[1].set_ylabel("Energy [eV]")
            ax[1].set_xlabel("Step Number in Structure Optimization")
            ax[0].imshow(i)
            ax[1].plot(E, label="energy [eV]")
            ax[1].legend()
            ax[1].plot([c, c], [np.min(E), np.max(E)], '-k')
            plt.tight_layout()
            plt.savefig("_tmp/tmp.jpg")
            writer.append_data(imageio.v2.imread("_tmp/tmp.jpg"))
        writer.close()
