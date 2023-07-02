import jax
import jax.numpy as jnp
import numpy as np
import pyscf
from exchange_correlation.b3lyp import b3lyp
from electron_repulsion.direct import prepare_integrals_2_inputs, compute_integrals_2, ipu_direct_mult, prepare_ipu_direct_mult_inputs
jax.config.FLAGS.jax_platform_name = 'cpu'
hartree_to_eV    = 27.2114079527 

def nanoGPT_iteration(i, vals):
    mask, V_xc, density_matrix, _V, _H, mf_diis_H, vj, vk, overlap, electron_repulsion, \
        fixed_hamiltonian, L_inv, weights, hyb, ao, num_calls, iter_matrices, part_energies = vals
    N = density_matrix.shape[0]

    # Step 1: Build Hamiltonian
    hamiltonian = fixed_hamiltonian + V_xc
    sdf         = overlap @ density_matrix @ hamiltonian
    hamiltonian, _V, _H, mf_diis_H = DIIS(i, sdf, hamiltonian, _V, _H, mf_diis_H)

    # Step 2: Solve eigh (L_inv turns generalized eigh into eigh). 
    eigvects = L_inv.T @ _eigh(L_inv @ hamiltonian @ L_inv.T)[1] 

    # Step 3: Use result from eigenproblem to build new density matrix.
    eigvects           = eigvects * mask.reshape(1, -1) # the same as "eigvects[:, :n_electrons_half]"
    density_matrix     = 2 * eigvects @ eigvects.T
    E_xc, V_xc, vj, vk = xc( density_matrix, ao, electron_repulsion, weights, vj, vk, hyb, num_calls)

    # Tensors used to compute energy/hlgap on CPU in float64. 
    part_energies = part_energies.at[i].set(  E_xc )
    iter_matrices = jax.lax.dynamic_update_slice(iter_matrices, jnp.stack((density_matrix, vj, vk, hamiltonian)).reshape(1, 4, N, N), (i, 0, 0, 0))

    return [mask, V_xc, density_matrix, _V, _H, mf_diis_H, vj, vk, overlap, electron_repulsion, fixed_hamiltonian, L_inv, weights, hyb, ao, num_calls, iter_matrices, part_energies]

def xc(density_matrix, ao, electron_repulsion, weights, vj, vk, hyb, num_calls):
    assert args.xc == "b3lyp"
    n = density_matrix.shape[0]

    ao0dm = ao[0] @ density_matrix
    rho   = jnp.sum((ao0dm.reshape(1, -1, n)) * ao , axis=2)
    rho   = jnp.concatenate([jnp.clip(rho[:1], CLIP_RHO_MIN, CLIP_RHO_MAX), rho[1:4]*2])

    E_xc, vrho, vgamma = b3lyp(rho, EPSILON_B3LYP) 
    E_xc = jnp.sum( rho[0] * weights * E_xc ) 

    weird_rho = (jnp.concatenate([vrho.reshape(1, -1)*.5, 2*vgamma*rho[1:4]], axis=0) * weights ) 

    n, p = weird_rho.shape
    V_xc = jnp.sum( (ao * weird_rho.reshape(n, p, 1)), axis=0)
    V_xc      = ao[0].T @ V_xc
    V_xc      = V_xc + V_xc.T

    d = num_calls.size

    if args.backend == "ipu":
        _tuple_indices, _tuple_do_lists, _N = prepare_ipu_direct_mult_inputs(num_calls.size , mol)
        vj, vk = jax.jit(ipu_direct_mult, backend="ipu", static_argnums=(2,3,4,5,6,7,8,9))( electron_repulsion, density_matrix, _tuple_indices, _tuple_do_lists, _N, num_calls.size, tuple(args.indxs.tolist()), tuple(args.indxs.tolist()), int(args.threads), v=int(args.multv)) 
        vk = vk*hyb
    else:
        d = density_matrix.shape[0]
        E = electron_repulsion
        vj = jnp.sum(E.reshape(d**2, d**2) * density_matrix.reshape(1, -1), axis=1).reshape(d,d)
        vk = jnp.sum(E.transpose(1,2,0,3).reshape(d**2, d**2) * density_matrix.reshape(1, -1), axis=1).reshape(d,d)*jnp.asarray(hyb, dtype=E.dtype)

    V_xc    = V_xc + vj - vk/2

    return E_xc, V_xc, vj, vk


def DIIS(i, sdf, hamiltonian, _V, _H, mf_diis_H):
    # Update hamiltonian as linear combination of previous iterations
    mf_diis_head      = i % _V.shape[0]
    nd, d             = _V.shape

    errvec = (sdf - sdf.T)

    # Store current (hamiltonian,errvec) as flattened as row inside _V and _H.
    _V = jax.lax.dynamic_update_slice(_V, errvec.reshape(1, d),      (mf_diis_head, 0))
    _H = jax.lax.dynamic_update_slice(_H, hamiltonian.reshape(1, d), (mf_diis_head, 0))

    tmps = (_V.reshape(nd, 1, d) @ errvec.reshape(1, d, 1))
    tmps = tmps.reshape(-1)

    # Shapes in initial code depended on min(i, _V.shape[0]).
    # To allow jax.jit, we always use nd=_V.shape[0] and zero out
    # the additional stuff with the following mask.
    mask = jnp.where(np.arange(_V.shape[0]) < jnp.minimum(i+1, _V.shape[0]),       
                        jnp.ones(_V.shape[0], dtype=_V.dtype), jnp.zeros(_V.shape[0], dtype=_V.dtype))
    tmps = tmps * mask

    # Assign tmp into row/col 'mf_diis_head+1' of mf_diis_H
    mf_diis_H = jax.lax.dynamic_update_slice( mf_diis_H, tmps.reshape(1, -1), (mf_diis_head+1, 1) )
    mf_diis_H = jax.lax.dynamic_update_slice( mf_diis_H, tmps.reshape(-1, 1), (1, mf_diis_head+1) )

    # Compute new hamiltonian as linear combination of previous 8.
    # Coefficients are computed as pseudo_inverse of mf_diis_H.
    # The first 8 iterations we are constructing mf_diis_H so it has shape (2,2), (3,3), (4,4), ...
    # To allow jax.jit we pad to (9, 9) and just zero out the additional stuff...
    mask_            = jnp.concatenate([jnp.ones(1, dtype=mask.dtype), mask])                                    
    masked_mf_diis_H = mf_diis_H[:nd+1, :nd+1] * mask_.reshape(-1, 1) * mask_.reshape(1, -1)

    if args.backend == "ipu":  c = pinv0( masked_mf_diis_H )
    else:                      c = jnp.linalg.pinv(masked_mf_diis_H)[0, :] 

    scaled_H         = _H[:nd] * c[1:].reshape(nd, 1)
    hamiltonian      = jnp.sum( scaled_H, axis=0 ).reshape(hamiltonian.shape)

    return hamiltonian, _V, _H, mf_diis_H 


def _nanoGPT(density_matrix, kinetic, nuclear, overlap, ao, 
                electron_repulsion, weights, 
                mf_diis_space, N, hyb, mask, _input_floats, _input_ints, L_inv=None):
    # --- INITIALIZE MATRICES USED FOR DIIS --- #
    mf_diis_H       = np.zeros((mf_diis_space+1, mf_diis_space+1))
    mf_diis_H[0,1:] = mf_diis_H[1:,0] = 1
    mf_diis_H       = np.array(mf_diis_H)

    _V = np.zeros((mf_diis_space, N**2))
    _H = np.zeros((mf_diis_space, N**2))

    iter_matrices = np.zeros((args.its, 4, N, N))
    part_energies = np.zeros(args.its)

    fixed_hamiltonian = kinetic + nuclear

    # Initialize values before main compute.
    vj, vk, V_xc = [np.zeros(fixed_hamiltonian.shape) for _ in range(3)]

    if args.backend == "ipu" and not args.ipumult:
        _, _, _tuple_ijkl, _shapes, _sizes, _counts, indxs, indxs_inv, num_calls = prepare_integrals_2_inputs(mol)
        args.indxs = indxs
        electron_repulsion = compute_integrals_2( _input_floats, _input_ints, _tuple_ijkl, _shapes, _sizes, _counts, tuple(indxs_inv), num_threads=args.threads_int, v=args.intv)[0]
        electron_repulsion = [a  for a in electron_repulsion]
        print("bytes: ", np.sum([a.nbytes for a in electron_repulsion])/ 10**6) 
    else: 
        num_calls = electron_repulsion.shape[0]

    _num_calls = np.zeros(num_calls)
    _, V_xc, vj, vk = xc( density_matrix, ao, electron_repulsion, weights, vj, vk, hyb, _num_calls)

    vals = jax.lax.fori_loop(0, args.its, nanoGPT_iteration, [mask, V_xc, density_matrix, _V, _H, mf_diis_H, vj, vk, overlap, electron_repulsion,  
                                                             fixed_hamiltonian, L_inv, weights, hyb, ao, _num_calls, iter_matrices, part_energies])
    iter_matrices = vals[-2]
    part_energies = vals[-1]

    return iter_matrices, fixed_hamiltonian, part_energies, L_inv 

def nanoDFT(str, mf_diis_space=9):
    mol = pyscf.gto.mole.Mole()
    mol.build(atom=str, unit="Angstrom", basis=args.basis, spin=args.spin, verbose=0)

    n_electrons       = mol.nelectron
    n_electrons_half  = n_electrons//2
    N                 = mol.nao_nr()
    nuclear_energy    = mol.energy_nuc()                                                      
    hyb               = pyscf.dft.libxc.hybrid_coeff(args.xc, mol.spin)                       

    mask = np.ones(N)
    mask[n_electrons_half:] = 0

    # Initialize grid.
    grids            = pyscf.dft.gen_grid.Grids(mol)
    grids.level      = args.level 
    grids.build()
    weights          = grids.weights                                                         
    ao               = mol.eval_gto('GTOval_cart_deriv1' if mol.cart else 'GTOval_sph_deriv1', grids.coords, 4)

    # Initialize all (N, N) sized matrices. 
    density_matrix  = pyscf.scf.hf.init_guess_by_minao(mol).reshape(N, N) 
    kinetic         = mol.intor_symmetric('int1e_kin'). reshape(N, N)  
    nuclear         = mol.intor_symmetric('int1e_nuc'). reshape(N, N)
    overlap         = mol.intor_symmetric('int1e_ovlp').reshape(N, N)

    if args.backend == "cpu" or args.ipumult: electron_repulsion = mol.intor("int2e_sph")

    # Turns generalized eigenproblem into eigenproblem.
    fixed_hamiltonian   = kinetic + nuclear
    L_inv               = np.linalg.inv(np.linalg.cholesky(overlap.astype(np.float64)))

    input_floats, input_ints = prepare_integrals_2_inputs(mol)[:2]

    vals = jax.jit(_nanoGPT, static_argnums=(7,8))( density_matrix, kinetic, nuclear, overlap, ao, electron_repulsion, weights, 
                                                    mf_diis_space, N, hyb , mask, input_floats, input_ints, L_inv)

    iter_matrices, fixed_hamiltonian, part_energies, _  = [np.asarray(a).astype(np.float64) for a in vals]
    density_matrices, vjs, vks, hamiltonians            = [iter_matrices[:, i] for i in range(4)]

    mo_occ = np.concatenate([np.ones(n_electrons_half)*2, np.zeros(N-n_electrons_half)])
    lumo   = np.argmin(np.array(mo_occ))
    homo   = lumo - 1
    d      = L_inv.shape[0]

    energies = np.zeros(args.its)
    hlgaps   = np.zeros(args.its)
    for i in range(args.its):
        density_matrix = density_matrices[i]
        E_coulomb   = np.sum( density_matrix * vjs[i]) * .5
        E_xc        = part_energies[i] - np.sum(density_matrix * vks[i]) * 0.25
        energies[i] = np.sum(fixed_hamiltonian * density_matrices[i]) + E_coulomb + nuclear_energy + E_xc

        mo_energy   = np.linalg.eigh(L_inv @ hamiltonians[-1].reshape(d, d) @ L_inv.T)[0]
        hlgaps[i]   = np.abs( mo_energy[homo] - mo_energy[lumo] )

    energies, hlgaps   = [a * hartree_to_eV for a in [energies, hlgaps]] 
    return energies, hlgaps

def _eigh(x):
    if args.backend == "ipu":
        from jax_ipu_research.tile import ipu_eigh
        n = x.shape[0]
        pad = n % 2
        print(x.dtype)
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
    vals, vect = _eigh ( a )
    c = vect @ ( jnp.where( jnp.abs(vals) > cond, 1/vals, 0) * vect[0, :]) 
    return c

if __name__ == "__main__":  
    import argparse
    parser = argparse.ArgumentParser(description='Arguments for Density Functional Theory. ')
    parser.add_argument('-skipdiis',  action="store_true", help='Whether to skip DIIS; useful for benchmarking.')
    parser.add_argument('-num',       default=10,          type=int,   help='Run the first "num" test molecules. ')
    parser.add_argument('-id',        default=126,          type=int,   help='Run only test molecule "id". ')
    parser.add_argument('-its',       default=20,          type=int,   help='Number of Kohn-Sham iterations. ')
    parser.add_argument('-step',      default=1,           type=int,   help='If running 1000s of test cases, do molecules[args.skip::args.step]]')
    parser.add_argument('-spin',      default=0,           type=int,   help='Even or odd number of electrons? Currently only support spin=0')
    parser.add_argument('-str',       default="",          help='Molecule string, e.g., "H 0 0 0; H 0 0 1; O 1 0 0; "')
    parser.add_argument('-numerror', action="store_true",     help='Save all tensors to debug numerical errors. ')
    parser.add_argument('-ipumult', action="store_true",     help='On IPU do mult using full tensor ERI computed using PySCF (and not our Rys Quadrature implementation). ')
    parser.add_argument('-float32',   action="store_true", help='Whether to use float32 (default is float64). ')
    parser.add_argument('-basis',     default="STO-3G",    help='Which basis set to use. ')
    parser.add_argument('-xc',        default="b3lyp",     help='Only support B3LYP. ')
    parser.add_argument('-backend',   default="cpu",       help='Which backend to use, e.g., -backend cpu or -backend ipu')

    parser.add_argument('-level',    default=2, help="Level of the grids used by us (default=2). ", type=int)
    parser.add_argument('-plevel',   default=2, help="Level of the grids used by pyscf (default=2). ", type=int)
    parser.add_argument('-gdb',        default=-1, type=int,  help='Which version of GDP to load {10, 11, 13, 17}. ')
    parser.add_argument('-multv',    default=2, type=int, help='Which version of our einsum algorithm to use;comptues ERI@flat(v). Different versions trades-off for memory vs sequentiality. ')
    parser.add_argument('-intv',    default=1, type=int, help='Which version to use of our integral algorithm. ')

    parser.add_argument('-jit',  action="store_true")
    parser.add_argument('-profile',  action="store_true", help="Stops script in generation mode after one molecule; useful when using popvision to profile for -backend ipu")
    parser.add_argument('-pyscf',  action="store_true", help="Used to compute with reference implementation. ")
    parser.add_argument('-uniform_pyscf',  default = -1, type=float, help="Use reference implementation PySCF if 'np.random.uniform(0,1,(1))<args.uniform_pyscf'")
    parser.add_argument('-threads',  default=1, type=int, help="Number of threads to use to compute ipu_mult_direct. ")
    parser.add_argument('-threads_int',  default=1, type=int, help="Number of threads to use to do int2e_sph, accepts {1,...,6}. ")
    parser.add_argument('-split',  default=[1, 16], type=int, nargs="+", help='How to split during data generation over multiple POD16s. 7 47 means split dataset into 47 chunks and this IPU handles chunk 7 (zero indexed).')
    parser.add_argument('-limit', default=-1, type=int, help='smiles = args.smiles[:limit]; gdb molecules are sorted by hydrogens, this allows us to take the ones with fewer hydrogens for which DFT is faster. ')
    parser.add_argument('-gname',  default="", type=str, help='Folder name to store generate dataset; useful when using multiple pods to generate. ')
    parser.add_argument('-geneigh',  action="store_true" , help='Use generalized eigendecomposition like pyscf; relies on scipy, only works in debug mode with -forloop. ')

    args = parser.parse_args()

    import sys
    print(sys.argv)
    from natsort import natsorted
    print(natsorted(vars(args).items()) )

    print("[BASIS]", args.basis)

    sys.argv = sys.argv[:1]

    if args.numerror:
        args.forloop = True
        args.backend = "cpu"
        args.its = 35

    print("")

    if args.backend == "ipu":  
        args.float32 = True

    if args.float32:
        EPSILON_B3LYP  = 1e-20
        CLIP_RHO_MIN   = 1e-9
        CLIP_RHO_MAX   = 1e12

    else:  # float64
        jax.config.update('jax_enable_x64', True) 
        EPSILON_B3LYP  = 1e-20
        CLIP_RHO_MIN   = 1e-9
        CLIP_RHO_MAX   = 1e12

    backend = args.backend
    eigh = _eigh

    if args.gdb > 0:
        if args.gdb == 10: args.smiles = [a for a in open("gdb/gdb11_size10_sorted.csv", "r").read().split("\n")]
        if args.gdb == 9:  args.smiles = [a for a in open("gdb/gdb11_size09_sorted.csv", "r").read().split("\n")]
        if args.gdb == 7:  args.smiles = [a for a in open("gdb/gdb11_size07_sorted.csv", "r").read().split("\n")]
        if args.gdb == 8:  args.smiles = [a for a in open("gdb/gdb11_size08_sorted.csv", "r").read().split("\n")]
        if args.gdb == 6:  args.smiles = ["c1ccccc1"]*1000 
        if args.gdb == 5:  args.smiles = ['CCCCC']*1000
        if args.gdb == 4:  args.smiles = ['CCCC']*1000
        if args.gdb == 3:  args.smiles = ['CCC']*1000
        if args.gdb == 2:  args.smiles = ['CC']*1000

        print("Length GDB: ", len(args.smiles))

        from rdkit import Chem 
        from rdkit.Chem import AllChem
        from rdkit import RDLogger
        lg = RDLogger.logger()
        lg.setLevel(RDLogger.CRITICAL)


        for i in range(0, len(args.smiles)): 
            smile = args.smiles[i]
            smile = smile

            b = Chem.MolFromSmiles(smile)
            b = Chem.AddHs(b)  
            atoms = [atom.GetSymbol() for atom in b.GetAtoms()]

            e = AllChem.EmbedMolecule(b) 
            if e == -1: continue

            locs = b.GetConformer().GetPositions() #* angstrom_to_bohr
            def get_atom_string(atoms, locs):
                import re
                atom_string = atoms
                atoms = re.findall('[a-zA-Z][^A-Z]*', atoms)
                str = ""
                for atom, loc in zip(atoms, locs):
                    str += "%s %4f %4f %4f; "%((atom,) + tuple(loc) )
                return atom_string, str

            atom_string, string = get_atom_string(" ".join(atoms), locs)

            print(string)
            break

        args.str = string 

    # Test Case: Compare nanoGPT against PySCF. 
    molecule_string = args.str
    mol = pyscf.gto.mole.Mole()
    mol.build(atom=molecule_string, unit="Angstrom", basis=args.basis, spin=args.spin, verbose=0)

    print("\t", molecule_string, end="")

    energies, hlgaps = nanoDFT(molecule_string)

    mol = pyscf.gto.mole.Mole()
    mol.verbose = 0
    pyscf.__config__.dft_rks_RKS_grids_level = args.plevel
    mol.build(atom=molecule_string, unit='Angstrom', basis=args.basis, spin=args.spin)

    mol.max_cycle = args.its
    mf = pyscf.scf.RKS(mol)
    mf.xc = args.xc

    if args.skipdiis:
        print("\n[ TURNING OFF DIIS IN PYSCF ]")
        mf.diis = 0
        mf.diis_space = 0
    mf.diis_space = 9 

    pyscf_energy    = mf.kernel()  * hartree_to_eV
    lumo           = np.argmin(mf.mo_occ)
    homo           = lumo - 1
    hl_gap_hartree = np.abs(mf.mo_energy[homo] - mf.mo_energy[lumo])

    print("pyscf_hlgap\t%15f"%( hl_gap_hartree * hartree_to_eV))
    print("us_hlgap\t%15f"%(    hlgaps[-1]))
    print("err_hlgap\t%15f"%np.abs((hl_gap_hartree * hartree_to_eV) - hlgaps[-1]))
    print("pyscf:\t\t%15f"%pyscf_energy)
    print("us:\t\t%15f"%energies[-1])
    print("mus:\t\t%15f"%np.mean(energies[-10:]))
    print("diff:\t\t%15f"%np.abs(pyscf_energy-energies[-1]))
    print("mdiff:\t\t%15f"%np.abs(pyscf_energy-np.mean(energies[-10:])), np.std(energies[-10:])) 
    print("chemAcc: \t%15f"%0.043)
    print("chemAcc/diff: \t%15f"%(0.043/np.abs(pyscf_energy-energies[-1])))
    print("chemAcc/mdiff: \t%15f"%(0.043/np.abs(pyscf_energy-np.mean(energies[-10:]))))
    print("> diffs:")
    print(np.abs(energies.reshape(-1)[-5:] - pyscf_energy))
    print("mean+-var: %f +- %f"%( np.mean(np.abs(energies.reshape(-1)[-5:] - pyscf_energy)), np.var(np.abs(energies.reshape(-1)[-5:] - pyscf_energy))))
