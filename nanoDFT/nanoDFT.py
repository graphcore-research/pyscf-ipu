# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import jax
import jax.numpy as jnp
import numpy as np
import pyscf
from jsonargparse import CLI, Namespace
import sys
sys.path.append("../")
from exchange_correlation.b3lyp import b3lyp
from electron_repulsion.direct import prepare_electron_repulsion_integrals, electron_repulsion_integrals, ipu_einsum

HARTREE_TO_EV = 27.2114079527
EPSILON_B3LYP = 1e-20
HYB_B3LYP = 0.2


def energy(density_matrix, H_core, J, K, E_xc, E_nuc, _np=jax.numpy):
    """Density Functional Theory (DFT) solves the optimisation problem:

        min_{density_matrix} energy(density_matrix, ...)

    We like to think of `energy(...)` as a loss function. `density_matrix`
    represents the density of electrons as: 

        rho(r) = sum_{ij}^N density_matrix_{ij} X_i(r) X_j(r)  where X_i(r)~exp(-r^2).

    Here N is the number of atomic orbitals (AO) **and** molecular orbitals (N=66 for C6H6). 
    All input matrices (density_matrix, H_core, J, K) are (N, N). The X_i(r) are called
    Gaussian Type Orbitals (GTO). The inputs (J, K, E_xc) depend on density_matrix. 
    """
    E_core = _np.sum(density_matrix * H_core)                   # float = -712.04[Ha] for C6H6.
    E_J    = _np.sum(density_matrix * J)                        # float =  624.38[Ha] for C6H6.
    E_K    = _np.sum(density_matrix * K)                        # float =   26.53[Ha] for C6H6.

    E      = E_core + E_J/2 - E_K/4 + E_xc + E_nuc              # float = -232.04[Ha] for C6H6.

    return _np.array([E, E_core, E_J/2, -E_K/4, E_xc, E_nuc])   # Energy (and its terms). 


def nanoDFT_iteration(i, vals): 
    """Each call updates density_matrix attempting to minimize energy(density_matrix, ... ). """ 
    density_matrix, V_xc, J, K, O, H_core, L_inv                    = vals[:7]                  # All (N, N) matrices
    E_nuc, occupancy, ERI, grid_weights, grid_AO, diis_history, log = vals[7:]                  # Varying types/shapes. 

    # Step 1: Update Hamiltonian (optionally use DIIS to improve DFT convergence). 
    H = H_core + J - K/2 + V_xc                                                                 # (N, N)  
    if args.diis: H, diis_history = DIIS(i, H, density_matrix, O, diis_history)                 # H_{i+1}=c_1H_i+...+c9H_{i-9}.

    # Step 2: Solve eigh (L_inv turns generalized eigh into eigh).
    eigvects = L_inv.T @ linalg_eigh(L_inv @ H @ L_inv.T)[1]                                    # (N, N) 

    # Step 3: Use result from eigenproblem to update density_matrix. 
    density_matrix = (eigvects*occupancy*2) @ eigvects.T                                        # (N, N)
    E_xc, V_xc     = exchange_correlation(density_matrix, grid_AO, grid_weights)                # float (N, N)
    J, K           = get_JK(density_matrix, ERI)                                                # (N, N) (N, N)

    # Log SCF matrices and energies (not used by DFT algorithm). 
    log["matrices"] = log["matrices"].at[i].set(jnp.stack((density_matrix, J, K, H)))           # (iterations, 4, N, N)
    log["energy"] = log["energy"].at[i].set(energy(density_matrix, H_core, J, K, E_xc, E_nuc))  # (iterations, 6)

    return [density_matrix, V_xc, J, K, O, H_core, L_inv, E_nuc, occupancy, ERI, grid_weights, grid_AO, diis_history, log]


def exchange_correlation(density_matrix, grid_AO, grid_weights):
    """Compute exchange correlation integral using atomic orbitals (AO) evalauted on a grid. """
    rho = jnp.sum(grid_AO[:1] @ density_matrix * grid_AO, axis=2)                                # (4, grid_size)=(4, 45624) for C6H6.
    E_xc, vrho, vgamma = b3lyp(rho, EPSILON_B3LYP)                                               # (gridsize,) (gridsize,) (gridsize,)          
    E_xc = jnp.sum(rho[0] * grid_weights * E_xc)                                                 # float=-27.968[Ha] for C6H6 at convergence.

    rho = jnp.concatenate([vrho.reshape(1, -1)/2, 4*vgamma*rho[1:4]], axis=0) * grid_weights     # (4, grid_size)=(4, 45624)
    V_xc = grid_AO[0].T @ jnp.sum(grid_AO * jnp.expand_dims(rho, axis=2), axis=0)                # (N, N)
    V_xc = V_xc + V_xc.T                                                                         # (N, N)

    return E_xc, V_xc                                                                            # (float) (N, N) 


def get_JK(density_matrix, ERI):                
    """Computes the (N, N) matrices J and K. Density matrix is (N, N) and ERI is (N, N, N, N).  """
    if args.backend != "ipu":
        J = jnp.einsum('ijkl,ji->kl', ERI, density_matrix)                                       # (N, N) 
        K = jnp.einsum('ijkl,jk->il', ERI, density_matrix)                                       # (N, N)
    else: 
        # Custom einsum which utilize ERI[ijkl]=ERI[ijlk]=ERI[jikl]=ERI[jilk]=ERI[lkij]=ERI[lkji]=ERI[lkij]=ERI[lkji]
        J, K = ipu_einsum(ERI, density_matrix, mol, args.threads, args.multv)                    # (N, N) (N, N)
    return J, K * HYB_B3LYP                                                                      # (N, N) (N, N)


def make_jitted_nanoDFT(backend): return jax.jit(_nanoDFT, backend=backend)
def _nanoDFT(E_nuc, density_matrix, kinetic, nuclear, O, grid_AO, ERI, grid_weights, 
              mask, _input_floats, _input_ints, L_inv, diis_history):
    # Utilize the IPUs MIMD parallism to compute the electron repulsion integrals (ERIs) in parallel. 
    if args.backend == "ipu": ERI = electron_repulsion_integrals(_input_floats, _input_ints, mol, args.threads_int, args.intv)
    else: pass # Compute on CPU. 
    
    # Precompute the remaining tensors.
    E_xc, V_xc = exchange_correlation(density_matrix, grid_AO, grid_weights) # float (N, N) 
    J, K       = get_JK(density_matrix, ERI)                                 # (N, N) (N, N) 
    H_core     = kinetic + nuclear                                           # (N, N)

    # Log matrices from all DFT iterations (not used by DFT algorithm). 
    N = H_core.shape[0]
    log = {"matrices": np.zeros((args.its, 4, N, N)), "E_xc": np.zeros((args.its)), "energy": np.zeros((args.its, 6))}

    # Perform DFT iterations. 
    log = jax.lax.fori_loop(0, args.its, nanoDFT_iteration, [density_matrix, V_xc, J, K, O, H_core, L_inv,  # all (N, N) matrices 
                                                            E_nuc, mask, ERI, grid_weights, grid_AO, diis_history, log])[-1]

    return log["matrices"], H_core, log["energy"]


def init_dft_tensors_cpu(args, DIIS_iters=9):
    mol = pyscf.gto.mole.Mole()
    mol.build(atom=args.mol_str, unit="Angstrom", basis=args.basis, verbose=0)
    N                = mol.nao_nr()                                 # N=66 for C6H6 (number of atomic **and** molecular orbitals)
    n_electrons_half = mol.nelectron//2                             # 21 for C6H6 
    E_nuc            = mol.energy_nuc()                             # float = 202.4065 [Hartree] for C6H6. TODO(): Port to jax.

    # TODO(): port grid/eval_gto to Jax. 
    grids            = pyscf.dft.gen_grid.Grids(mol)            
    grids.level      = args.level
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
    if args.backend != "ipu": ERI = mol.intor("int2e_sph")          # (N,N,N,N)=(66,66,66,66) for C6H6.
    else:                     ERI = None # will be computed on device

    input_floats, input_ints = prepare_electron_repulsion_integrals(mol)[:2]
    mask = np.concatenate([np.ones(n_electrons_half), np.zeros(N-n_electrons_half)])

    # DIIS is an optional technique to improve DFT convergence. 
    DIIS_H       = np.zeros((DIIS_iters+1, DIIS_iters+1))
    DIIS_H[0,1:] = DIIS_H[1:,0] = 1
    diis_history = (np.zeros((DIIS_iters, N**2)), np.zeros((DIIS_iters, N**2)), DIIS_H)

    tensors = (E_nuc, density_matrix, kinetic, nuclear, O, grid_AO, ERI, 
               grid_weights, mask, input_floats, input_ints, L_inv, diis_history)

    return tensors, n_electrons_half, E_nuc, N, L_inv 


def nanoDFT(args):
    # Init DFT tensors on CPU using PySCF. 
    tensors, n_electrons_half, E_nuc, N, L_inv = init_dft_tensors_cpu(args)

    # Run DFT algorithm (can be hardware accelerated). 
    vals = jitted_nanoDFT(*tensors)  
    logged_matrices, H_core, logged_energies = [np.asarray(a).astype(np.float64) for a in vals] # Ensure CPU 

    # It's cheap to compute energy/hlgap on CPU in float64 from the logged values/matrices. 
    logged_E_xc = logged_energies[:, 4].copy()
    density_matrices, Js, Ks, H = [logged_matrices[:, i] for i in range(4)]
    energies, hlgaps = np.zeros((args.its, 6)), np.zeros(args.its)
    for i in range(args.its):
        energies[i] = energy(density_matrices[i], H_core, Js[i], Ks[i], logged_E_xc[i], E_nuc, np)
        hlgaps[i]   = hlgap(L_inv, H[i], n_electrons_half, np)
    energies, logged_energies, hlgaps = [a * HARTREE_TO_EV for a in [energies, logged_energies, hlgaps]] 
    return energies, logged_energies, hlgaps


def DIIS(i, H, density_matrix, O, diis_history):
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

    if args.backend == "ipu":  c = pinv0(masked_DIIS_H)                       # (diis_iters+1,)=10 for C6H6.
    else:                      c = jnp.linalg.pinv(masked_DIIS_H)[0, :]       # (diis_iters+1,)=10 for C6H6.

    H = (c[1:] @ _H).reshape(H.shape)                                         # (N, N)
    return H, (_V, _H, DIIS_H)                                                # (N, N)


def hlgap(L_inv, H, n_electrons_half, _np):
    mo_energy   = _np.linalg.eigh(L_inv @ H @ L_inv.T)[0]
    return _np.abs(mo_energy[n_electrons_half] - mo_energy[n_electrons_half-1])


def linalg_eigh(x):
    if args.backend == "ipu":
        from jax_ipu_experimental_addons.tile import ipu_eigh
        n = x.shape[0]
        pad = n % 2
        if pad:
            x = jnp.pad(x, [(0, 1), (0, 1)], mode='constant')

        eigvects, eigvals = ipu_eigh(x, sort_eigenvalues=True, num_iters=12)

        if pad:
            e1 = eigvects[-1:]
            col = jnp.argmax(e1)
            eigvects = jnp.roll(eigvects, -col-1)
            eigvects = eigvects[:, :-1]
            eigvects = jnp.roll(eigvects, -(-col))
            eigvects = eigvects[:-1]
    else:
        eigvals, eigvects = jnp.linalg.eigh(x)

    return eigvals, eigvects


def pinv0(a):  # take out first row
    cond =  9*1.1920929e-07
    vals, vect = linalg_eigh ( a )
    c = vect @ ( jnp.where( jnp.abs(vals) > cond, 1/vals, 0) * vect[0, :])
    return c


def pyscf_reference(args):
    mol = pyscf.gto.mole.Mole()
    mol.verbose = 0
    pyscf.__config__.dft_rks_RKS_grids_level = args.level
    mol.build(atom=args.mol_str, unit='Angstrom', basis=args.basis, spin=0)

    mol.max_cycle = args.its
    mf = pyscf.scf.RKS(mol)
    mf.max_cycle = args.its
    mf.xc = args.xc
    mf.diis_space = 9 
    if not args.diis:  # 
        mf.diis_space = 0 
        mf.diis = False 
    pyscf_energies = []
    pyscf_hlgaps = [] 
    lumo         = mol.nelectron//2 
    homo         = lumo - 1
    def callback(envs): # (TODO) compute different energy terms (XC, kin, ...) and compare to nanoDFT. 
        pyscf_energies.append(envs["e_tot"]*HARTREE_TO_EV)
        hl_gap_hartree = np.abs(envs["mo_energy"][homo] - envs["mo_energy"][lumo]) * HARTREE_TO_EV
        pyscf_hlgaps.append(hl_gap_hartree)
    mf.callback = callback
    mf.kernel()  
    return np.array(pyscf_energies), np.array(pyscf_hlgaps)


def print_difference(energies, logged_energies, hlgaps, pyscf_energies, pyscf_hlgaps):
    #TODO(HH): rename to match caller variable names
    print("pyscf_hlgap\t%15f"%( pyscf_hlgaps[-1]))
    print("us_hlgap\t%15f"%(    hlgaps[-1]))
    print("err_hlgap\t%15f"%np.abs(pyscf_hlgaps[-1]  - hlgaps[-1]))
    print("pyscf:\t\t%15f"%pyscf_energies[-1])
    print("us:\t\t%15f"%energies[-1, 0])
    print("mus:\t\t%15f"%np.mean(energies[-10:, 0]))
    print("diff:\t\t%15f"%np.abs(pyscf_energies[-1]-energies[-1, 0]))
    print("mdiff:\t\t%15f"%np.abs(pyscf_energies[-1]-np.mean(energies[-10:, 0])), np.std(energies[-10:, 0]))
    print("chemAcc: \t%15f"%0.043)
    print("chemAcc/diff: \t%15f"%(0.043/np.abs(pyscf_energies[-1]-energies[-1, 0])))
    print("chemAcc/mdiff: \t%15f"%(0.043/np.abs(pyscf_energies[-1]-np.mean(energies[-10:, 0]))))
    print("")
    pyscf_energies = np.concatenate([pyscf_energies, np.ones(energies.shape[0]-pyscf_energies.shape[0])*pyscf_energies[-1]])  
    pyscf_hlgaps = np.concatenate([pyscf_hlgaps, np.ones(hlgaps.shape[0]-pyscf_hlgaps.shape[0])*pyscf_hlgaps[-1]])  
    print("%18s"%"", "\t".join(["%10s"%str("iter %i "%i) for i in np.arange(1, energies.shape[0]+1)[1::3]]))
    print("%18s"%"Error Energy [eV]", "\t".join(["%10s"%str("%.2e"%f) for f in (pyscf_energies[1::3] - energies[1::3, 0]).reshape(-1)]))
    print("%18s"%"Error HLGAP [eV]", "\t".join(["%10s"%str("%.2e"%f) for f in (pyscf_hlgaps[1::3]   - hlgaps[1::3]).reshape(-1)]))
 
    # E_core, E_J/2, -E_K/4, E_xc, E_nuc
    print()
    print("%18s"%"E_core [eV]", "\t".join(["%10s"%str("%.5f"%f) for f in (energies[1::3, 1]).reshape(-1)]))
    print("%18s"%"E_J [eV]", "\t".join(["%10s"%str("%.5f"%f) for f in (energies[1::3, 2]).reshape(-1)]))
    print("%18s"%"E_K [eV]", "\t".join(["%10s"%str("%.5f"%f) for f in (energies[1::3, 3]).reshape(-1)]))
    print("%18s"%"E_xc [eV]", "\t".join(["%10s"%str("%.5f"%f) for f in (energies[1::3, 4]).reshape(-1)]))
    print("%18s"%"E_nuc [eV]", "\t".join(["%10s"%str("%.5f"%f) for f in (energies[1::3, 5]).reshape(-1)]))

    
def nanoDFT_parser(
        its: int = 20,
        mol_str: str = "benzene",
        float32: bool = False,
        basis: str = "6-31G",
        xc: str = "b3lyp",
        backend: str = "cpu",
        level: int = 1,
        multv: int = 2,
        intv: int = 1,
        threads: int = 1,
        threads_int: int = 1,
        diis: bool = True, 
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
        mol_str = "C        0.0000    0.0000    0.0000; C        1.4000    0.0000    0.0000; C        2.1000    1.2124    0.0000; C        1.4000    2.4249    0.0000; C        0.0000    2.4249    0.0000; C       -0.7000    1.2124    0.0000; H       -0.5500   -0.9526    0.0000; H       -0.5500    3.3775    0.0000; H        1.9500   -0.9526    0.0000; H       -1.8000    1.2124    0.0000; H        3.2000    1.2124    0.0000; H        1.9500    3.3775    0.0000;"
    elif mol_str == "methane":
        mol_str = "C 0 0 0; H 0 0 1; H 0 1 0; H 1 0 0; H 1 1 1;"
    args = locals()
    args = Namespace(**args)
    jax.config.update('jax_enable_x64', not float32)
    return args


if __name__ == "__main__":
    # Limit PySCF threads to mitigate problem with NUMA nodes. 
    import os
    os.environ['OMP_NUM_THREADS'] = "8"
    args = CLI(nanoDFT_parser)
    assert args.xc == "b3lyp"

    jitted_nanoDFT = make_jitted_nanoDFT(args.backend) # used later inside nanoDFT 

    # Test Case: Compare nanoDFT against PySCF.
    mol = pyscf.gto.mole.Mole()
    mol.build(atom=args.mol_str, unit="Angstrom", basis=args.basis, spin=0, verbose=0)

    nanoDFT_E, nanoDFT_logged_E, nanoDFT_hlgap = nanoDFT(args)
    pyscf_E, pyscf_hlgap = pyscf_reference(args)
    print_difference(nanoDFT_E, nanoDFT_logged_E, nanoDFT_hlgap, pyscf_E, pyscf_hlgap)
