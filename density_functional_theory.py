import jax
import jax.numpy as jnp 
from jax.config import config
config.FLAGS.jax_platform_name = 'cpu'
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
from tqdm import tqdm 
import scipy 
import os
import pyscf 
from pyscf import gto, scf
from pyscf import __config__ 
import argparse
from natsort import natsorted
import sys 
from rdkit import Chem  
from rdkit.Chem import AllChem
import csv 
import numpy as np 
import pyscf.dft
from pyscf.scf import hf
from pyscf.gto.mole import Mole
from pyscf import scf 
from pyscf import gto 
import time 
import pandas as pd 
import re 
from pyscf_utils.minao      import minao 
from pyscf_utils.build_grid import build_grid
#from pyscf_utils.fast_grid import build_grid
from pyscf_utils.build_mol  import build_mol
from exchange_correlation.b3lyp import b3lyp
from exchange_correlation.b3lyp import do_lda as lda 
from rdkit import Chem # 
from rdkit.Chem import AllChem
from rdkit import RDLogger
from electron_repulsion.direct import prepare_int_floats, prepare_integrals_2_inputs
from electron_repulsion.direct import prepare_integrals_2_inputs, compute_integrals_2, ipu_direct_mult, prepare_ipu_direct_mult_inputs
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

angstrom_to_bohr = 1.88973
hartree_to_eV    = 27.2114

def get_atom_string(atoms, locs):
    atom_string = atoms 
    atoms = re.findall('[a-zA-Z][^A-Z]*', atoms)
    str = ""
    for atom, loc in zip(atoms, locs): 
      str += "%s %4f %4f %4f; "%((atom,) + tuple(loc) )
    return atom_string, str 

def _do_compute(density_matrix, kinetic, nuclear, overlap, ao, #L_inv, 
                electron_repulsion, weights, coords, nuclear_energy, disable_cache, mf_diis_space, N, hyb, mask, _input_floats, _input_ints, L_inv=None): 
                # coords is not used in here, only to produce weights and ao. 
        # --- INITIALIZE MATRICES USED FOR DIIS --- # 
        #print("COMPILING", mf_diis_space)

        mf_diis_H       = np.zeros((mf_diis_space+1, mf_diis_space+1))              
        mf_diis_H[0,1:] = mf_diis_H[1:,0] = 1                                                                
        mf_diis_H       = np.array(mf_diis_H)

        # Stores last 8 hamiltonians/error_vectors, used to compute "momentum-like" term. 
        _V = np.zeros((mf_diis_space, N**2))                                                       
        _H = np.zeros((mf_diis_space, N**2))                                                        

        dms = np.zeros((args.its, 3, N, N))
        part_energies = np.zeros(args.its)

        fixed_hamiltonian = kinetic + nuclear
        
        overlap = overlap 

        if L_inv is None: 
            cholesky = jnp.linalg.cholesky(overlap) 
            L_inv    = jnp.linalg.pinv(cholesky) 
        else: 
            cholesky = jnp.zeros(overlap.shape)

        eigvals, eigvects, energy = np.zeros(fixed_hamiltonian.shape[0]), np.zeros(fixed_hamiltonian.shape), 0. 
        energies = np.zeros(args.its)  

        # Initialize values before main compute. 
        vj, vk, V_xc = [np.zeros(fixed_hamiltonian.shape) for _ in range(3)]
        eigvals, eigvects, energy = np.zeros(fixed_hamiltonian.shape[0]), np.zeros(fixed_hamiltonian.shape), 0. 
        energies = np.zeros(args.its)  
        cs = np.zeros((args.its, mf_diis_space+1))
        allvals = np.zeros((args.its, density_matrix.shape[0]))

        if args.backend == "ipu" and not args.ipumult:
            from electron_repulsion.direct import prepare_integrals_2_inputs , compute_integrals_2 
            _, _, _tuple_ijkl, _shapes, _sizes, _counts, indxs, indxs_inv, num_calls = prepare_integrals_2_inputs(mol)
            args.indxs = indxs 
            if not args.seperate: 
                #electron_repulsion, cycles_start, cycles_stop = compute_integrals_2( _input_floats, _input_ints, _tuple_ijkl, _shapes, _sizes, _counts, tuple(indxs_inv), num_threads=args.threads_int, v=args.intv)
                electron_repulsion = compute_integrals_2( _input_floats, _input_ints, _tuple_ijkl, _shapes, _sizes, _counts, tuple(indxs_inv), num_threads=args.threads_int, v=args.intv)[0]
                electron_repulsion = [a  for a in electron_repulsion] 
            print("bytes: ", np.sum([a.nbytes for a in electron_repulsion])/ 10**6) # ~ 5MB for gdb11  (so 5*8=40 MB normally). 
        elif not args.seperate: 
            #print(electron_repulsion)
            num_calls = electron_repulsion.shape[0]

        if args.skiperi: electron_repulsion = 1.
        elif args.randeri: 
            print("Generating random eri")
            t0 = time.time()
            electron_repulsion = np.empty((N, N, N,N)) 
            print(time.time()-t0)
        if args.float16: electron_repulsion = electron_repulsion.astype(np.float16)

        generalized_hamiltonian = jnp.zeros(overlap.shape)
        hamiltonian = jnp.zeros(overlap.shape)
        sdf = jnp.zeros(overlap.shape)
        errvec = jnp.zeros(overlap.shape)

        # should this go first or last? or both? 
        # initial xc? 
        _num_calls = np.zeros(num_calls)
        cycle = 0 # this computes initial V_xc 
        if type(electron_repulsion) == type([]):
            if args.float32: E_xc, V_xc, E_coulomb, vj, vk = xc( density_matrix.astype(np.float32), dms.astype(np.float32), cycle, ao.astype(np.float32), electron_repulsion, weights.astype(np.float32), vj.astype(np.float32), vk.astype(np.float32), hyb, _num_calls)
        else: 
            if args.float32: E_xc, V_xc, E_coulomb, vj, vk = xc( density_matrix.astype(np.float32), dms.astype(np.float32), cycle, ao.astype(np.float32), electron_repulsion.astype(np.float32), weights.astype(np.float32), vj.astype(np.float32), vk.astype(np.float32), hyb, _num_calls)
            else: E_xc, V_xc, E_coulomb, vj, vk = xc( density_matrix.astype(np.float64), dms.astype(np.float64), cycle, ao.astype(np.float64), electron_repulsion.astype(np.float64), weights.astype(np.float64), vj.astype(np.float64), vk.astype(np.float64), hyb, _num_calls)

        #if type(part_energies) == type(jnp.array(1)): part_energies = part_energies.at[cycle].set(  E_xc ) 
        #else: part_energies[cycle] =E_xc  
        vals = [mask, allvals, cs, energies, V_xc, density_matrix, _V, _H, mf_diis_H, 
                    vj, vk, eigvals, eigvects, energy, overlap, electron_repulsion, 
                fixed_hamiltonian, L_inv, weights, hyb, ao, nuclear_energy, _num_calls, cholesky, generalized_hamiltonian, sdf, errvec, hamiltonian, dms, part_energies]

        # storing the numbers may be the issue? 
        #vals = [f(a, False) for a in vals]
        vals = [f(a, args.float32) for a in vals]

        # unwrap for loop is faster? 
        if args.nan or args.forloop: 
            if args.jit: _iter = jax.jit(iter, backend=args.backend)

            os.makedirs("experiments/numerror/%s/"%(str(args.sk)) , exist_ok=True)
            for n in tqdm(range(int(args.its))): 

                if args.jit: vals = _iter(n, vals)# 
                else: vals = iter(n, vals) 

                if args.numerror: 
                    _str = ["mask", "allvals", "cs", "energies", "V_xc", "density_matrix", "_V", "_H", "mf_diis_H", 
                            "vj", "vk", "eigvals", "eigvects", "energy", "overlap", "electron_repulsion", 
                            "fixed_hamiltonian", "L_inv", "weights", "hyb", "ao", "nuclear_energy", "np.zeros(num_calls)", "cholesky", "generalized_hamiltonian", "sdf", "errvec", "hamiltonian", "dms", "part_energies"]
                    for s, v  in zip(_str, vals):
                        
                        #np.savez("experiments/numerror/%s/%i_%s_%s.npz"%(str(args.sk), n, s, args.float32), v=v)
                        np.savez("experiments/numerror/%i_%s_%s.npz"%( n, s, args.float32), v=v)

        elif args.its == 1: 
            vals = jax.jit(iter, backend=args.backend)(0, vals)
        else:
            #print("[jax.lax.fori_loop]")
            vals = jax.lax.fori_loop(0, args.its, iter, vals) 

        #vals = [mask, allvals, cs, energies, V_xc, density_matrix, _V, _H, mf_diis_H, 
        #         vj, vk, eigvals, eigvects, energy, overlap, electron_repulsion, 
        #        fixed_hamiltonian, L_inv, weights, hyb, ao, nuclear_energy, np.zeros(num_calls), 
        #       cholesky, generalized_hamiltonian, sdf, errvec, hamiltonian, dms, part_energies]

        #vals = jax.jit(iter, backend="cpu")
        eigenvalues   = vals[11]
        eigenvectors  = vals[12]
        energy        = vals[13]
        energies      = vals[ 3]
        dms           = vals[28]
        part_energies = vals[29]

        # set dms for last iteration;; 
        #density_matrix, vj, hamiltonian = vals[4], vals[9], vals[-3]
        #dms                        = dms.at[-1, 0].set(density_matrix)#.T.reshape(-1)) 
        #dms                        = dms.at[-1, 1].set(vj)#.reshape(-1)) 
        #dms                        = dms.at[-1, 2].set(hamiltonian)#.reshape(-1)) 

        return energies, energy, eigenvalues, eigenvectors, dms, fixed_hamiltonian, part_energies, L_inv #, vals
        # in this last iteration, is it fast enough to do everything on CPU, except doing ERI@x? 
        #return vals[:14] + vals[15:]  # return everything but electron repulsion (this will take too long to return, too big!)
        #return vals 

def density_functional_theory(atom_positions, mf_diis_space=9):                              # Example values for benchmark molecule (C6H6 Benzene)
    if args.backend == "ipu": mf_diis_space = 9                                              # remove cause padding works!  put back in because it doens't to eigvals which are neede for pinv! 

    if args.generate: 
        global mol 


    # Numbers dependent on molecule. 
    nuclear_energy    = mol.energy_nuc()                                                      # 204.58834163716134
    n_electrons       = mol.nelectron                                                         # 42
    n_electrons_half  = n_electrons//2                                                        # 21
    hyb               = pyscf.dft.libxc.hybrid_coeff(args.xc, mol.spin)                       # 0.2
    N                 = mol.nao_nr()                                                          # 96 (NAO = number of atomic orbitals)

    mask = np.ones(N)
    mask[n_electrons_half:] = 0 
    if args.float32: 
        mask = mask.astype(np.float32)

    # Initialize grid.
    grids            = pyscf.dft.gen_grid.Grids(mol) 
    grids.level      = args.level # getattr(pyscf.__config__, 'dft_rks_RKS_grids_level', grids.level-args.level) 
    grids.build()     
    coords           = grids.coords                                                           # (143560, 3)=(grid_size, 3d) [[ -4.98003649 -12.07183349  -9.74340349], ...]
    weights          = grids.weights                                                          #* args.scale_w# (143560, ) =(grid_size, )   [0.00536925 0.00103828 0.00154942 0.02549841 0.00155558]
    weights          = weights 
    ao              = mol.eval_gto('GTOval_cart_deriv1' if mol.cart else 'GTOval_sph_deriv1', coords, 4) 

    # Initialize all (N, N) sized matrices (using PySCF or Jax depending on args). 
    density_matrix  = np.array(minao(mol)).reshape(N, N)  #/ 100 # this doesn't change anything =O

    # Don't see theese called in pyscf only libcint; #pragma omp appears many times in pyscf but only in 2 files in libcint
    kinetic         = mol.intor_symmetric('int1e_kin'). reshape(N, N)  # might be in libcint and thus not use threads? 
    nuclear         = mol.intor_symmetric('int1e_nuc'). reshape(N, N) 
    overlap         = mol.intor_symmetric('int1e_ovlp').reshape(N, N) 

    print(weights.shape, ao.shape, coords.shape)
    print(weights.astype(np.float32).nbytes/10**6, ao.astype(np.float32).nbytes/10**6, coords.astype(np.float32).nbytes/10**6)
    #exit()

    if (args.backend == "cpu" or args.ipumult) and not args.seperate:
        electron_repulsion = mol.intor("int2e_sph") 
    else: 
        electron_repulsion = 0. 

    # Turns generalized eigenproblem into eigenproblem. 
    fixed_hamiltonian   = kinetic + nuclear 
    #L_inv               = np.linalg.inv(np.linalg.cholesky(overlap.astype(np.float64)))
    L_inv               = np.linalg.inv(np.linalg.cholesky(overlap))#.astype(np.float64)))

    if args.generate:  
        __config__.dft_rks_RKS_grids_level = args.plevel 

        #rng = range(10**6) # this is tricky? 
        rng = range(len(args.smiles)) # this is tricky? 
        if args.gdb > 0:
            id, num = args.split 
            num_mols = len(args.smiles) // num
            print(len(args.smiles))
            rng  = list(range(id*num_mols, (id+1)*num_mols))
            print(">>> ", min(rng), max(rng))

        if args.save: 
            suffix = "_%i_%i"%(min(rng), max(rng))
            if args.fname != "":
                # check if a folder already exists, if so, the folder already (partially) exists
                # this may kill the log files... perhaps add a timestamp to the log file? 
                os.makedirs("data/generated/%s/"%(args.fname), exist_ok=True)

                folders = os.listdir("data/generated/%s/"%args.fname)
                # do we have one with the specific split? 
                print("_%i_%i"%(min(rng), max(rng)) )

                # fix code for generating. 

                done = []
                for folder in folders:
                    if "_%i_%i"%(min(rng), max(rng)) in folder: 
                        print("HIT")
                        print(folder)
                        file = "data/generated/%s/%s/data.csv"%(args.fname, folder)
                        if os.path.isfile(file): 
                            df = pd.read_csv(file, compression="gzip")
                            print(df.shape)
                            done.append(df)
                        else:
                            print("NO!")

                if len(done)>0 and args.resume: 
                  df = pd.concat(done)

                  all_smiles = pd.DataFrame( args.smiles[min(rng):max(rng)] )
                  done_smiles = df["smile"]
                  print(all_smiles.shape)
                  print(done_smiles.shape)

                  not_done = all_smiles[~ all_smiles[0].isin(done_smiles)]
                  args.smiles = not_done.values
                  print(args.smiles[:10])
                  args.smiles = [item for sublist in args.smiles for item in sublist] # makes list of lists into flattened list. 

                  print("Left: ", len(args.smiles))
                  #print(not_done.shape)


                  # this breaks the next loading...
                  rng = range(len(args.smiles)) #  todo: simplify this by removing rng from loop below by removing from args.smiles according to rng. 


                name = "%i_GDB%i_f32%s_grid%i_backend%s"%(len(os.listdir("data/generated/%s/"%args.fname)), int(args.gdb), args.float32, args.level, args.backend)
                name += suffix 
                print(name)
                os.makedirs("data/generated/%s/%s/"%(args.fname, name), exist_ok=True)


            else: 
                name = "%i_GDB%i_f32%s_grid%i_backend%s"%(len(os.listdir("data/generated/")), int(args.gdb), args.float32, args.level, args.backend)
                name += "_%i_%i"%(min(rng), max(rng))
                print(name)
                os.makedirs("data/generated/%s/"%name, exist_ok=True)

        assert args.plevel == args.level 
        initialized = False 

        device_1 = jax.devices(args.backend)[0]
        #device_1 = jax.devices(args.backend)[0]
        function = jax.jit(_do_compute, device=device_1, static_argnums=(10,11)) 
        if args.seperate: 
            device_2 = jax.devices("ipu")[1]
            print(device_2)
            compute_integrals = jax.jit(compute_integrals_2, static_argnums=(2,3,4,5,6,7), device=device_2)  

        pbar = tqdm(rng)
        init_int = True

        # figure out largest grid size by looking at last molecule. 
        if True: 
            for j in range(1, len(args.smiles)):
                i = rng[-j] # start with hardest? why did I do this? 

                smile = args.smiles[i]
                #smile = "CCC#CC=CC(C)=O"# "NCCNC=CC(N)=N"
                atoms = [a for a in list(smile.upper()) if a == "C" or a == "N" or a == "O" or a == "F"]

                b = Chem.MolFromSmiles(smile)
                #b = Chem.AddHs(b, explicitOnly=False)   
                if not args.nohs: b = Chem.AddHs(b, explicitOnly=False)   # AddExplici,tHs AddPolarHss

                AllChem.EmbedMolecule(b)

                atoms = [atom.GetSymbol() for atom in b.GetAtoms()]
                num_hs = len([a for a in atoms if a == "H"])
                try: 
                    locs =  b.GetConformer().GetPositions() * angstrom_to_bohr  
                except:
                    continue 

                atom_string, string = get_atom_string(" ".join(atoms), locs)

                mol = Mole() 
                mol = build_mol(mol, atom=string, unit="Bohr", basis=args.basis, spin=args.spin, verbose=0)

                # Initialize grid. Depends on molecule! 
                grids            = pyscf.dft.gen_grid.Grids(mol)  # 40ms
                grids.level      = args.level 
                grids            = build_grid(grids) # custom version of grids.build() 
                coords           = grids.coords                                                           
                weights          = grids.weights 
                pad = int(weights.shape[0]*1.1) # assuming 10% larger than last molecule is ok. 
                print("[PAD] Last molecule had grisize=%i we're using %i. "%(weights.shape[0], pad))
                break # if we didn't skip break 
 
        np.random.seed(42)
        embedded = 0
        not_embedded = 0

        vals  = []
        times = [] 

        # Add to make for loop over 10 conformers for each molecule => 10x dataset! 
        # Try to do 100 conformers => 0.4M => 40M for gdb9 
        ptable = Chem.GetPeriodicTable()
        for count, i in enumerate(pbar): 
            
            # Having this first is nice because it allows continue to start preparing the next molecule. 
            if True: 

                if args.gdb == 1337: 
                    conformers = [args.qm9["pos"].values[i][0]]*3 # 
                    atoms = [ptable.GetElementSymbol( n) for n in args.qm9["z"].values[i][0].tolist() ]
                    #print(conformers, atoms)
                    num_hs = len([a for a in atoms if a == "H"])
                    #print(">>>>>>>>>", args.qm9[i:i+1])

                else: 
                    times.append(time.perf_counter())   #3
                    smile = args.smiles[i]
                    print("[%s]"%smile)
                    atoms = [a for a in list(smile.upper()) if a == "C" or a == "N" or a == "O" or a == "F"]

                    b = Chem.MolFromSmiles(smile)

                    if not args.nohs: b = Chem.AddHs(b, explicitOnly=False)   # AddExplici,tHs AddPolarHss

                    #embed_result = AllChem.EmbedMolecule(b, maxAttempts=1, randomSeed=42)
                    #embed_result = AllChem.EmbedMultipleConfs(b, numConfs=args.num_conformers, randomSeed=42)
                    embed_result = AllChem.EmbedMultipleConfs(b, numConfs=args.num_conformers, randomSeed=43)
                    if embed_result == -1: 
                        not_embedded += 1
                        continue 
                    else: 
                        embedded += 1

                    conformers = [a.GetPositions() for a in b.GetConformers()]
                    print("[conformers]", len(conformers))

                    atoms = [atom.GetSymbol() for atom in b.GetAtoms()]
                    num_hs = len([a for a in atoms if a == "H"])

            finished_first_iteration  = False 
            vals = [] # removing this out might give us 1/500 faster? i.e., in first iteration of new conformer fix the previous one? 

            for conformer_num, conformer in enumerate(conformers): 

                #print("!!!")

                if vals != []:
                    prev_mol = mol 
                    old_nuclear_energy = nuclear_energy
                    prev_smile = smile 
                    prev_locs = locs 
                    prev_atoms = atoms 

                times.append(time.perf_counter()) # 3 
                #locs = conformer.GetPositions() * angstrom_to_bohr 
                locs = conformer * angstrom_to_bohr 
                if args.rattled_std != 0: 
                    print(locs)
                    locs += np.random.normal(0, args.rattled_std, locs.shape)
                    print(locs)

                times.append(time.perf_counter()) # 4
                atom_string, string = get_atom_string(" ".join(atoms), locs)
                times.append(time.perf_counter()) #5

                mol = Mole() 
                try: 
                    mol = build_mol(mol, atom=string, unit="Bohr", basis=args.basis, spin=args.spin, verbose=0)
                except: 
                    print("BROKEN!")
                    continue
                times.append(time.perf_counter()) 

                nuclear_energy   = mol.energy_nuc()                                                  
                n_electrons      = mol.nelectron                                                     
                n_electrons_half = n_electrons//2                                                   
                hyb              = pyscf.dft.libxc.hybrid_coeff(args.xc, mol.spin)                  
                N                = mol.nao_nr()                                                     

                mask = np.ones(N)
                mask[n_electrons_half:] = 0 

                times.append(time.perf_counter()) # 7

                if (args.backend == "cpu" or args.ipumult) and not args.seperate: electron_repulsion = mol.intor('int2e_sph').reshape(N, N, N, N)  
                else:                     electron_repulsion = 0 

                # Initialize grid. Depends on molecule! 
                if conformer_num == 0 or True:  
                    times.append(time.perf_counter()) # 8
                    grids            = pyscf.dft.gen_grid.Grids(mol)  # 40-80ms
                    grids.level      = args.level 
                    times.append(time.perf_counter()) # 170ms ; this is the g * g thingy that takes forever 
                    grids            = build_grid(grids) # custom version of grids.build() 
                times.append(time.perf_counter()) # 10
                coords          = grids.coords                                                           
                weights         = grids.weights 
                ao              = mol.eval_gto('GTOval_cart_deriv1' if mol.cart else 'GTOval_sph_deriv1', coords, 4)  
                times.append(time.perf_counter()) # 11 
                kinetic         = mol.intor_symmetric('int1e_kin'). reshape(N, N)  
                times.append(time.perf_counter()) # 12
                nuclear         = mol.intor_symmetric('int1e_nuc'). reshape(N, N)
                times.append(time.perf_counter()) # 13 
                overlap         = mol.intor_symmetric('int1e_ovlp').reshape(N, N)  
                times.append(time.perf_counter()) # 14 
                fixed_hamiltonian   = kinetic + nuclear 
                if vals != []: 
                    L_inv_prev = L_inv
                    homo_prev = homo 
                    lumo_prev = lumo 

                lumo = n_electrons_half
                homo = lumo - 1
                try: 
                    # 80 ms 
                    #L_inv_prev = L_inv ;; looks like computing L_inv on IPU is better? 
                    #L_inv = None #np.linalg.inv(np.linalg.cholesky(overlap.astype(np.float64)))
                    if args.choleskycpu: 
                        L_inv = np.linalg.inv(np.linalg.cholesky(overlap.astype(np.float64)))
                    else: 
                        L_inv = None 
                except:
                    print("MATRIX NOT POSITIVE DEFINITE, SKIPPING MOLECULE: ", smile)
                    continue 
                times.append(time.perf_counter()) # 15

                # Don't recompute minao every iteration. 
                # Check how different minao is. 
                if conformer_num == 0: init_density_matrix = hf.init_guess_by_minao(mol) 
                if np.sum(np.isnan(density_matrix)) > 0 or density_matrix.shape != kinetic.shape: density_matrix = hf.init_guess_by_minao(mol)
                density_matrix = init_density_matrix 
                times.append(time.perf_counter()) # 16 ;; this still takes 8 ms

                weights = np.pad(weights, (0, pad-weights.shape[0]))
                coords  = np.pad(coords, ((0, pad-weights.shape[0]), (0, 0)))
                ao      = np.pad(ao, ((0, 0), (0, pad-ao.shape[1]), (0, 0)))

                times.append(time.perf_counter()) # 17 
                _input_floats, _input_ints = prepare_int_floats(mol) 
                times.append(time.perf_counter()) # 18

                if args.backend != "cpu": # don't pad on cpu? 
                    if _input_floats.shape[0] != 400: 
                        _input_floats = np.concatenate((_input_floats, np.zeros((1, 400-_input_floats.shape[1]))), axis=1)
                
                if args.seperate: 
                    if init_int: _input_floats, _input_ints, _tuple_ijkl, _shapes, _sizes, _counts, indxs, indxs_inv, num_calls = prepare_integrals_2_inputs(mol)
                    init_int = False  # this runs every iteration, noo? 
                    if _input_floats.shape[0] != 400: 
                        _input_floats = np.concatenate((_input_floats, np.zeros((1, 400-_input_floats.shape[1]))), axis=1)
                    times.append(time.perf_counter())  
                    electron_repulsion, cycles_start, cycles_stop = compute_integrals( _input_floats, _input_ints, _tuple_ijkl, _shapes, _sizes, _counts, tuple(indxs_inv), args.threads_int)
                    times.append(time.perf_counter())  

                times.append(time.perf_counter())  #19

                if args.save and vals != []:   # this saves the last iteration for us! 
                    times.append(time.perf_counter())  # 20
                    energies_, energy, eigenvalues, eigenvectors, dms, _fixed_hamiltonian, part_energies, _L_inv_prev  = [np.asarray(a).astype(np.float64) for a in vals[-1]]
                    times.append(time.perf_counter())  #21

                    if not args.choleskycpu: L_inv_prev = _L_inv_prev # 

                    e = np.zeros(energies_.shape)
                    for i in range(energies_.shape[0]): 
                        density_matrix = dms[i,0].reshape(-1) # not transposed
                        vj             = dms[i,1].reshape(-1)  #not transposed
                        E_coulomb      = np.sum( (density_matrix) * vj) * .5 
                        e[i]           = part_energies[i] + np.dot(_fixed_hamiltonian.reshape(-1) , dms[i, 0].reshape(-1)) + E_coulomb + old_nuclear_energy

                    _N = int(np.sqrt(density_matrix.shape[0]))
                    density_matrix = density_matrix.reshape(_N, _N)
                    energy = np.mean(e[-5:])

                    times.append(time.perf_counter())  #21
                    # recall eigh(A @ B)=eigh(B @ A) => we could use inv(overlap) instead! 
                    # this may also apply to computation of the eigenvectors during the algorithm? 
                    # this can throw "eigenvalues not converged"

                    # TODO: check whether hlgap is different for different conformers?
                    try: 
                        #mo_energy_us = np.linalg.eigh(L_inv_prev @ dms[-1, 2].reshape(_N,_N) @ L_inv_prev.T)[0] 
                        mo_energy_us = np.linalg.eigvalsh(L_inv_prev @ dms[-1, 2].reshape(_N,_N) @ L_inv_prev.T) 
                    except: 
                        try:
                            mo_energy_us = np.linalg.eigh(L_inv_prev @ dms[-1, 2].reshape(_N,_N) @ L_inv_prev.T)[0] 
                            #mo_energy_us = np.linalg.eigvalsh(L_inv_prev @ dms[-1, 2].reshape(_N,_N) @ L_inv_prev.T) 
                        except: 
                            continue 
                    
                    # run pyscf on prev molecule 
                    pyscf_energy = 0
                    pyscf_energies = np.zeros(1)
                    pyscf_homo = 0
                    pyscf_lumo = 0
                    pyscf_hlgap = 0
                    rand = np.random.uniform(0,1,(1)) 
                    if args.pyscf or rand < args.uniform_pyscf: 
                        print("\n\nRUNNING PYSCF\n\n")

                        pyscf_energies = []
                        pyscf_dms = []
                        def callback(envs):
                            energy = envs['e_tot'] 
                            pyscf_energies.append(energy)
                            pyscf_dms.append(envs["dm"])

                        prev_mol.max_cycle = args.its 
                        mf = scf.RKS(prev_mol) 
                        mf.callback = callback
                        mf.xc = args.xc 

                        mf.diis_space = mf_diis_space 

                        if args.skipdiis: 
                            print("\n[ TURNING OFF DIIS IN PYSCF ]")
                            mf.diis = 0 
                            mf.diis_space = 0 

                        t0 = time.time() 
                        pyscf_energy    = mf.kernel()  
                        print("[pyscf time]", time.time()-t0, pyscf_energy*hartree_to_eV)
                        pyscf_time = time.time()-t0
                        #print(pyscf_energies) 

                        pyscf_homo = mf.mo_energy[homo]*hartree_to_eV
                        pyscf_lumo = mf.mo_energy[lumo]*hartree_to_eV
                        pyscf_hlgap = np.abs(pyscf_homo - pyscf_lumo )

                        pyscf_energies = np.array(pyscf_energies)

                        if args.pyscf: 
                            np.set_printoptions(suppress=True)
                            print(pyscf_energy*hartree_to_eV)
                            print(0.034)

                    times.append(time.perf_counter()) #2
                    #mo_energy_us = np.linalg.eigvalsh(dms[-1, 2].reshape(_N,_N) @ L_inv_prev.T@L_inv_prev)
                    #mo_energy_us = scipy.linalg.eigh(dms[-1:, 2], axis=0).reshape(_N,_N) , overlap_pre, eigvals_only = True)
                    times.append(time.perf_counter())  #21
                    us_homo = mo_energy_us[homo_prev]*hartree_to_eV
                    us_lumo = mo_energy_us[lumo_prev]*hartree_to_eV
                    us_hlgap = np.abs(us_homo - us_lumo)#*hartree_to_eV
                    #print("us_hlgap", us_hlgap)
                    if args.verbose: 
                        print("jaxdft_hlgap\t", us_hlgap)
                        print("pyscf_hlgap\t", pyscf_hlgap )
                        print("error\t\t", np.abs(us_hlgap-pyscf_hlgap))

                        print(energy*hartree_to_eV)
                        print(pyscf_energy*hartree_to_eV)
                        print(e[-5:]*hartree_to_eV)
                        print(np.array(pyscf_energies[-5:])*hartree_to_eV)

                    if args.checkc and conformer_num > 0: 

                        # Add the hlgaps aswell. 
                        hlgaps_us = []
                        hlgaps_pyscf = []
                        dm_diff = []

                        fig, ax = plt.subplots(1,3, figsize=(10,4))
                        energies = e.reshape(-1)*hartree_to_eV
                        ax[0].plot(energies, label="energy us")
                        ax[0].plot(pyscf_energies*hartree_to_eV, label="energy pyscf")
                        ax[0].legend()
                        #plt.ticklabel_format(style='plain', axis='y')
                        _e = np.concatenate([np.array(pyscf_energies), np.ones(energies.shape[0]-len(pyscf_energies))*pyscf_energies[-1]])*hartree_to_eV
                        ax[1].plot(np.abs(energies-_e), label="|us-pyscf|")
                        ax[1].plot([0, args.its], [np.abs(energies[-1]-np.mean(_e[-10:])), np.abs(energies[-1]-np.mean(_e[-10:]))], label="|maen(us[-10:])-pyscf|")
                        ax[1].plot([0, args.its], [np.abs(energies[-1]-np.median(_e[-10:])), np.abs(energies[-1]-np.median(_e[-10:]))], label="|median(us[-10:]-pyscf|")
                        ax[1].legend()
                        ax[1].set_yscale("log")
                        ax[1].set_ylim([1e-10, 1e5])
                        ax[1].set_xlabel("iterations")
                        ax[1].set_ylabel("absolute |us-pyscf| (energy eV)")


                        ax[2].plot(np.abs(energies-_e)/np.abs(_e), label="|us-pyscf|/|pyscf|")
                        ax[2].legend()
                        ax[2].set_yscale("log")
                        ax[2].set_ylim([1e-10, 1e5])
                        ax[2].set_xlabel("iterations")
                        ax[2].set_ylabel("relative |us-pyscf|/|pyscf| (energy eV)")

                        if args.float32: plt.suptitle("float32")
                        else: plt.suptitle("float64")

                        print("[checkc %s]"%args.backend)

                        plt.tight_layout()
                        plt.savefig("experiments/checkc/%i_%i_f32_%s.jpg"%(count, conformer_num, str(args.float32)))
                        np.savez("experiments/checkc/%i_%i_%s_%s_%s.npz"%(count, conformer_num, str(args.float32), args.backend, smile), pyscf=pyscf_energies*hartree_to_eV, us=e)

                    if args.fname != "":
                        filename = "data/generated/%s/%s/data.csv"%(args.fname, name)
                    else: 
                        filename = "data/generated/%s/data.csv"%(name)

                    if args.gdb: 
                        timez = np.array(times)
                        timez = np.around((timez[1:] - timez[:-1])*1000, 1)
                        dct = { # we are storing smile strnigs "off-by-one"
                                "smile":          prev_smile, 
                                "atoms":          "".join(prev_atoms), 
                                "atom_positions": [prev_locs.reshape(-1).tolist()], 
                                "energies":       [(e.reshape(-1)*hartree_to_eV).tolist()], 
                                "std":            np.std(e.reshape(-1)[-5:]*hartree_to_eV),
                                "pyscf_energies": [np.array(pyscf_energies.reshape(-1)*hartree_to_eV).tolist()], 
                                "pyscf_hlgap":    pyscf_hlgap,
                                "pyscf_homo":     pyscf_homo,
                                "pyscf_lumo":     pyscf_lumo,
                                "times":          [timez],
                                "homo":           us_homo, 
                                "lumo":           us_lumo,
                                "hlgap":          us_hlgap,
                                "N":              prev_mol.nao_nr(),
                                "basis": args.basis,
                        }

                        
                        if args.pyscf: 
                            dct["pyscf_time"]   = pyscf_time 
                            dct["pyscf_energy"] = pyscf_energy*hartree_to_eV
                            dct["pyscf_homo"]   = pyscf_homo 
                            dct["pyscf_lumo"]   = pyscf_lumo
                            dct["pyscf_hlgap"]  = pyscf_hlgap

                        times.append(time.perf_counter()) #10
                        if not os.path.isfile(filename):  pd.DataFrame(dct).to_csv(filename, mode='a', header=True, compression="gzip")
                        else:  pd.DataFrame(dct).to_csv(filename, mode='a', header=False, compression="gzip")
                           
                if vals != []:
                    times.append(time.perf_counter()) #10
                    times = np.array(times)
                    times = np.around((times[1:] - times[:-1])*1000, 1)
                    pbar.set_description("[%i / %i] Hs=%5i "%(conformer_num, len(conformers), num_hs) +  "%10f "%energy + " ".join([str(a) for a in times.tolist() + [np.around(np.sum(times), 1)]])  + " [%i ; %i]"%(embedded, not_embedded))

                vals  = []
                times = [time.perf_counter()]  # 0 
                if np.sum(np.isnan(density_matrix)) > 0 or density_matrix.shape != kinetic.shape:
                    density_matrix  = np.array(minao(mol))
                if not args.forloop: 
                    if not args.skip_minao: density_matrix  = np.array(minao(mol)) # difference isn't caused by this! 
                    vals.append( function( density_matrix, kinetic, nuclear, overlap, ao, electron_repulsion, weights, coords, nuclear_energy, 0, mf_diis_space, N, hyb , mask, _input_floats, _input_ints, L_inv)  )
                else: 
                    # make this into a flag. 
                    if not args.skip_minao: density_matrix  = np.array(minao(mol)) # difference isn't caused by this! 
                    vals.append(_do_compute( density_matrix, kinetic, nuclear, overlap, ao, electron_repulsion, weights, coords, nuclear_energy, 0, mf_diis_space, N, hyb , mask, _input_floats, _input_ints, L_inv) )
    
                times.append(time.perf_counter())  #1
                

                if args.profile: 
                    exit()

                # so not running in generate works; 
                # potential explanations
                # - input wrong parameters to DFT in generate 
                # - generate compares molecule_i against molecule_i+1 for energy but not hlgap 


    c = 1
    if not args.forloop: 


        input_floats, input_ints, _tuple_ijkl, _shapes, _sizes, _counts, indxs, indxs_inv, num_calls = prepare_integrals_2_inputs(mol)
        print(num_calls)

        print(args.backend)
        device_1 = jax.devices(args.backend)[0]

        if args.seperate:
            device_2 = jax.devices("ipu")[1]
            compute_integrals = jax.jit(compute_integrals_2, static_argnums=(2,3,4,5,6,7,8), device=device_2)  
            electron_repulsion, cycles_start, cycles_stop = compute_integrals( input_floats, input_ints, _tuple_ijkl, _shapes, _sizes, _counts, tuple(indxs_inv), num_threads=int(args.threads_int)) 
            electron_repulsion = [np.asarray(a)  for a in electron_repulsion] 


        vals = jax.jit(_do_compute, static_argnums=(10,11), device=device_1) ( density_matrix, kinetic, nuclear, overlap, 
                                                                                       ao, electron_repulsion, weights, coords, nuclear_energy,
                                                                                       0,
                                                                                       mf_diis_space, N, hyb ,
                                                                                       mask, input_floats, input_ints, L_inv) 

        energies_, energy, eigenvalues, eigenvectors, dms, fixed_hamiltonian, part_energies, _  = [np.asarray(a).astype(np.float64) for a in vals]
        print(density_matrix.dtype)
        e = np.zeros(energies_.shape)
        for i in range(energies_.shape[0]):
            density_matrix = dms[i,0] # transposed
            vj = dms[i,1]  #not transposed
            E_coulomb = np.sum( (density_matrix/c) * vj) * .5 
            e[i] = part_energies[i] + np.dot(fixed_hamiltonian.reshape(-1) , dms[i, 0].reshape(-1))  + E_coulomb + nuclear_energy

        f32_energy = e[-1:]
     
        print(f32_energy*hartree_to_eV, e[-1]*hartree_to_eV)

    else: 

        input_floats, input_ints, _tuple_ijkl, _shapes, _sizes, _counts, indxs, indxs_inv, num_calls = prepare_integrals_2_inputs(mol)
        vals =  _do_compute( density_matrix, kinetic, nuclear, overlap, 
                                                                                       ao, electron_repulsion, weights, coords, nuclear_energy,
                                                                                       0,
                                                                                       mf_diis_space, N, hyb ,
                                                                                       mask, input_floats, input_ints, L_inv) 

        print([a.dtype for a in vals])
        energies_, energy, eigenvalues, eigenvectors, dms, fixed_hamiltonian, part_energies  = [np.asarray(a).astype(np.float64) for a in vals]
        e = np.zeros(energies_.shape)
        for i in range(energies_.shape[0]):
            density_matrix = dms[i,0] # transposed
            vj = dms[i,1]  #not transposed
            E_coulomb = np.sum( (density_matrix.astype(np.float64)/c) * vj) * .5 
            e[i] = part_energies[i] + np.dot(fixed_hamiltonian.reshape(-1) , dms[i, 0].reshape(-1))  + E_coulomb + nuclear_energy
            
    return e, energy, eigenvalues, eigenvectors, dms, L_inv


# utility functions for investigating float{64,32}
def f(x, float32):
    if not args.float32: return x  # in float64 mode just return x 

    if args.backend == "ipu": 
        try: 
            return x.astype(np.float32) # could (shouldn't) do a copy
        except:
            return x 

    if not args.float32: 
        if type(x) == type(.1): return x 
        return x.astype(np.float64)

    if type(x) != type(.1) and type(x) != type([]) and x.dtype == np.float64:
        print("!!!!!!!!", x.shape, x.dtype, "!!!!!!!!")

        if float32: return x.astype(np.float32)

    if not args.debug: return x
    if float32: return jnp.asarray(x).astype(jnp.float32)
    elif type(x) == type(jnp.zeros(1)): return jnp.asarray(x).astype(jnp.float64)
    else: return x 

def kahan_dot(x, y, sort=False): # more stable dot product; blows up compile time, rewrite to poplar vertex. 
    xy = y * x
    sum   = jnp.array(0.0 ,dtype=xy.dtype)
    error = jnp.array(0.0, dtype=xy.dtype) 
    if sort: xy = xy[jnp.argsort(jnp.abs(xy))]
    def body(i, val):
        xy, error, sum = val 
        prod = xy[i] - error
        temp = sum + prod
        error = (temp - sum) - prod
        sum = temp 
        return (xy, error, sum)
    xy, error, sum = jax.lax.fori_loop(np.zeros(1, dtype=np.int32), np.asarray(len(x), dtype=np.int32), body, (xy, error, sum))
    return sum

def iter( cycle, val ):  
    # i think the problem is just that the other one has the iterations switched, eigh => other stuff, not other stuff => eigh
    # so if we can zero out the first iteration that may work? 
    mask, allvals, cs, energies, V_xc, density_matrix, _V, _H, mf_diis_H, vj, vk, eigvals, eigvects, energy, overlap, electron_repulsion, \
        fixed_hamiltonian, L_inv, weights, hyb, ao, nuclear_energy, num_calls, cholesky, generalized_hamiltonian, sdf, errvec, hamiltonian, dms, part_energies = val 

    for i, a in enumerate(val): 
        try:
            if type(a) == type([]): 
                print([(b.shape, b.nbytes/10**6) for b in a])
            else: 
                if a.nbytes>1000000: print(a.nbytes/10**6, a.shape, i)
        except:
            pass 

    # so we do xc first; this may be ok in that we're just doin gthe first iteration, but, we're storing the energy 
    # perhaps the problem is we're storing the energy one ahead? i.e., using energy from this iteration but should use for next? 
    # oh lol, perhaps the initial values are just not ok for the integral.. 

    # we do xc stuff first! 
    
    # Step 1: Build Hamiltonian 
    d = True and args.float32
    sdf, hamiltonian, _V, _H, mf_diis_H, fixed_hamiltonian, V_xc, overlap, density_matrix, hamiltonian = f(sdf, d), f(hamiltonian, d), f(_V, d), f(_H,d), f(mf_diis_H, d), f(fixed_hamiltonian, d), f(V_xc, d), f(overlap, d), f(density_matrix, d), f(hamiltonian, d)
    hamiltonian                    = fixed_hamiltonian + V_xc 
    sdf                            = overlap @ density_matrix @ hamiltonian  
    sdf, hamiltonian, _V, _H, mf_diis_H, fixed_hamiltonian, V_xc, overlap, density_matrix, hamiltonian = f(sdf, d), f(hamiltonian, d), f(_V, d), f(_H,d), f(mf_diis_H, d), f(fixed_hamiltonian, d), f(V_xc, d), f(overlap, d), f(density_matrix, d), f(hamiltonian, d)
    if not args.skipdiis: 
        hamiltonian, _V, _H, mf_diis_H, errvec = DIIS(cycle, sdf, hamiltonian, _V, _H, mf_diis_H)
    d = args.float32 
    sdf, hamiltonian, _V, _H, mf_diis_H, fixed_hamiltonian, V_xc, overlap, density_matrix, hamiltonian = f(sdf, d), f(hamiltonian, d), f(_V, d), f(_H,d), f(mf_diis_H, d), f(fixed_hamiltonian, d), f(V_xc, d), f(overlap, d), f(density_matrix, d), f(hamiltonian, d)

    # Step 2: Solve (generalized) eigenproblem for Hamiltonian:     generalized_hamiltonian = L_inv @ hamiltonian @ L_inv.T                                           
    d = True and args.float32 
    cholesky, hamiltonian = f(cholesky, d), f(hamiltonian, d)
    L_inv = f(L_inv, d) 

    generalized_hamiltonian = L_inv @ hamiltonian @ L_inv.T   
    d = args.float32
    generalized_hamiltonian, hamiltonian, cholesky = f(generalized_hamiltonian, d), f(hamiltonian,d), f(cholesky, d)


    d = True and args.float32 
    generalized_hamiltonian = f(generalized_hamiltonian, d)
    if args.skipeigh: eigvals, eigvects = hamiltonian[0], hamiltonian
    else:             
        eigvects = _eigh(generalized_hamiltonian )[1]  # can scale arbitrarily

    d = True and args.float32 
    generalized_hamiltonian, cholesky = f(generalized_hamiltonian, d), f(cholesky, d)
    eigvects          = L_inv.T @ eigvects  
    if args.geneigh: 
        import scipy
        _, eigvects = scipy.linalg.eigh(hamiltonian, overlap) 
    d = args.float32
    cholesky, hamiltonian, eigvects = f(cholesky, d), f(hamiltonian, d), f(eigvects, d)

    # Step 3: Use result from eigenproblem to build new density matrix. 
    # Use masking instead of """eigvects[:, :n_electrons_half]""" to allow changing {C,O,N,F} without changing compute graph => compiling only once. 
    d = True and args.float32
    eigvects = f(eigvects, d)
    eigvects         =     eigvects * mask.reshape(1, -1)
    old_dm = density_matrix 
    density_matrix   = (2 * eigvects @ eigvects.T) 

    if args.density_mixing: 
        # skip first iteration or first 8 iterations 
        # do density mixing only after the initial DIIS thing has filled up vectors
        # looks like this just makes everything worse! 
        density_matrix = jax.lax.select(cycle<=_V.shape[0], density_matrix, (old_dm+density_matrix)/2)

    d = args.float32  
    density_matrix, eigvects = f(density_matrix, d), f(eigvects, d) 


    # should this go first or last? or both? 
    if type(electron_repulsion) == type([]):
        if args.float32: E_xc, V_xc, E_coulomb, vj, vk = xc( density_matrix.astype(np.float32), dms.astype(np.float32), cycle, ao.astype(np.float32), electron_repulsion, weights.astype(np.float32), vj.astype(np.float32), vk.astype(np.float32), hyb, num_calls)
    else: 
        if args.float32: E_xc, V_xc, E_coulomb, vj, vk = xc( density_matrix.astype(np.float32), dms.astype(np.float32), cycle, ao.astype(np.float32), electron_repulsion.astype(np.float32), weights.astype(np.float32), vj.astype(np.float32), vk.astype(np.float32), hyb, num_calls)
        else: E_xc, V_xc, E_coulomb, vj, vk = xc( density_matrix.astype(np.float64), dms.astype(np.float64), cycle, ao.astype(np.float64), electron_repulsion.astype(np.float64), weights.astype(np.float64), vj.astype(np.float64), vk.astype(np.float64), hyb, num_calls)

    if type(part_energies) == type(jnp.array(1)): part_energies = part_energies.at[cycle].set(  E_xc ) 
    else: part_energies[cycle] =E_xc  


    # Needed to add dynamic_update_slice for compiler not to go crazy. 
    if args.debug or True:  
        N = density_matrix.shape[0]
        if type(dms) == type(jnp.array(1)):
            dms = jax.lax.dynamic_update_slice(dms, density_matrix.reshape(1, 1, N, N),   (cycle, 0, 0, 0))  
            dms = jax.lax.dynamic_update_slice(dms, vj.reshape(1, 1, N, N),               (cycle, 1, 0, 0))  
            dms = jax.lax.dynamic_update_slice(dms, hamiltonian.reshape(1, 1, N, N),      (cycle, 2, 0, 0))  
        else: 
            dms[cycle, 0] = density_matrix.reshape(N,N)
            dms[cycle, 1] = vj.reshape(N,N)
            dms[cycle, 2] = hamiltonian.reshape(N,N)

    ret = [mask, allvals, cs, energies, V_xc, density_matrix, _V, _H, mf_diis_H, vj, vk, eigvals, eigvects, energy, overlap, electron_repulsion, fixed_hamiltonian, L_inv, weights, hyb, ao, nuclear_energy, num_calls, cholesky, generalized_hamiltonian, sdf, errvec, hamiltonian, dms, part_energies]

    split = 100
    ret = [f(a, args.float32) for a in ret[:split]] + [f(a, False) for a in ret[split:]]

    return ret

# alternatively, sort, and sum largest to smallest. 
def kahan_sum_sort(xy): # x is vector and y is matrix 
    # Initialize the sum and the error
    sum   = jnp.zeros((xy.shape[0], xy.shape[2]),dtype=xy.dtype)
    error = jnp.zeros((xy.shape[0], xy.shape[2]), dtype=xy.dtype)
    def body(i, val):
        xy, error, sum = val
        prod = xy[:, i, :] - error
        temp = sum + prod
        error = (temp - sum) - prod
        sum = temp 
        return (xy, error, sum)
    xy, error, sum = jax.lax.fori_loop(0, xy.shape[1], body, (xy, error, sum))
    return sum


def xc(density_matrix, dms, cycle, ao, electron_repulsion, weights, vj, vk, hyb, num_calls): 
    n = density_matrix.shape[0]

    # lesson: casting all of the things introduced error, which mislead into where the problems where. 
    # lesson: vj leads to more problems than vk; this is reflected in visualizations with vk in [1-4, 1e3] and vk [1e-15, 1e1e10] 


    switch = args.float32 

    d = 0 in args.sk 
    density_matrix, ao = f(density_matrix, d), f(ao, d)
    #ao0dm = ao[0] @ density_matrix 
    #ao0dm = kahan_sum( ao[0].reshape(-1, n, 1) * density_matrix.reshape(1, n, n))  # improved!  ~ 17 to 28  (89 with mDiff)
    if args.float32 and not args.backend == "ipu": 
        ao0dm = kahan_sum_sort(ao[0].reshape(-1, n, 1) * density_matrix.reshape(1, n, n) ) # this increases memory n**3
    else: 
        print("Matmul")
        ao0dm = ao[0] @ density_matrix  
    d = switch
    ao0dm, density_matrix, ao = f(ao0dm, d), f(density_matrix, d), f(ao, d)

    d = 1 in args.sk
    ao0dm, ao = f(ao0dm, d), f(ao, d)
    #rho                = jnp.einsum("np,inp->in", ao0dm, ao) /args.scale_ao / args.scale_ao 
    #rho               = jnp.sum((ao0dm.reshape(1, -1, n)) * ao , axis=2) /args.scale_ao / args.scale_ao  
    rho               = jnp.sum((ao0dm.reshape(1, -1, n)) * ao , axis=2) 

    #i,n,p = ao.shape
    #print(i,n,p, ao0dm.shape, ao.shape)
    #rho = ao.reshape(i,n,p) @ ao0dm.T / args.scale_ao / args.scale_ao
    #print(rho.shape)
    #rho               = jnp.sum((ao0dm.reshape(1, -1, n)) * (ao) , axis=2) /args.scale_ao / args.scale_ao 
    d = switch 
    rho, ao0dm, ao = f(rho, d), f(ao0dm, d), f(ao, d)

    d = 2 in args.sk  # 118.865857
    rho = f(rho, d)
    rho                = jnp.concatenate([jnp.clip(rho[:1], CLIP_RHO_MIN, CLIP_RHO_MAX), rho[1:4]*2])             
    d = switch 
    rho = f(rho, d)

    d = 3 in args.sk  # 82.188577
    rho = f(rho, d)
    if args.xc == "b3lyp": E_xc, vrho, vgamma = b3lyp(rho, EPSILON_B3LYP)                                                      
    elif args.xc == "lda": 
        print(rho.shape) # how big is rho? 
        E_xc, vrho, vgamma = lda(rho,   EPSILON_B3LYP)                                                      
        print("ASDJKASLD")
    else: E_xc, vrho, vgamma = b3lyp(rho, EPSILON_B3LYP)                                                      
    d = switch 
    E_xc, vrho, vgamma, rho = f(E_xc, d), f(vrho, d), f(vgamma, d), f(rho,d)

    d = 5 in args.sk   
    rho, weights, E_xc = f(rho, d), f(weights, d), f(E_xc, d)
    #E_xc =  (rho[0] * weights) @ E_xc / args.scale_w  
    E_xc =  jnp.sum( rho[0] * weights *  E_xc ) # <--- @weights 
    d = switch 
    rho, weights, E_xc = f(rho, d), f(weights, d), f(E_xc, d)

    d = 6 in args.sk  
    vrho, vgamma, rho, weights = f(vrho, d), f(vgamma, d), f(rho, d), f(weights, d) 
    weird_rho = (jnp.concatenate([vrho.reshape(1, -1)*.5, 2*vgamma*rho[1:4]], axis=0) * weights ) # <--- @weights
    d = switch 
    vrho, vgamma, rho, weights = f(vrho, d), f(vgamma, d), f(rho, d), f(weights, d) 

    d = 7 in args.sk   
    ao, weird_rho = f(ao, d), f(weird_rho, d) 
    #print(ao, weird_rho.shape)
    #V_xc      = jnp.einsum('npi,np->pi', ao[:4], weird_rho)/args.scale_ao 
    n, p = weird_rho.shape
    V_xc = jnp.sum( (ao * weird_rho.reshape(n, p, 1)), axis=0)
    #exit()
    d = switch 
    V_xc, ao, weird_rho = f(V_xc, d), f(ao, d), f(weird_rho, d)

    d = 8 in args.sk   
    ao, V_xc = f(ao, d), f(V_xc, d)
    V_xc      = ao[0].T @ V_xc 
    d = switch 
    ao, V_xc = f(ao, d), f(V_xc, d)

    d = 9 in args.sk #  56.137293
    V_xc = f(V_xc, d) 
    V_xc      = V_xc + V_xc.T
    d = switch
    V_xc = f(V_xc, d) 


    if not args.skiperi: # we're not scaling eri on ipu but doing on cpu. 
        d = num_calls.size#  electron_repulsion.shape[0]  # can remove 
        c = 1

        if args.backend == "ipu" and not args.ipumult or args.seperate:
            _tuple_indices, _tuple_do_lists, _N = prepare_ipu_direct_mult_inputs(num_calls.size , mol)

            ipu_vj, ipu_vk = jax.jit(ipu_direct_mult, backend="ipu", static_argnums=(2,3,4,5,6,7,8,9))(
                                                electron_repulsion, 
                                                density_matrix, 
                                                _tuple_indices,
                                                _tuple_do_lists, _N, num_calls.size,
                                                tuple(args.indxs.tolist()), # not used 
                                                tuple(args.indxs.tolist()), # we can impement Kahan dot in the poplar vertex! 
                                                int(args.threads),
                                                v=int(args.multv)
                                                ) # ipu mult;; we can make this compute V_xc also without much more flops! 

            vj         = ipu_vj
            vk         = ipu_vk*hyb

        else: 
            d = density_matrix.shape[0]
            #I = jnp.eye(d**2, dtype=density_matrix.dtype) 

            #c = 1
            c = 1
           
            d = 11 in args.sk  # what are things we can do here?  ;; scale_vj only changes here and not later 
            density_matrix, E = f(density_matrix, d), f(electron_repulsion, d), 

            d = density_matrix.shape[0]
            #vj_I = ((E.reshape(d**2, d**2)*c - I*c) @ density_matrix.reshape(d**2)).reshape(d,d) 
            #vj32 = vj_I - density_matrix.reshape(d, d) * c
            #vj32 = vj32/c
            #vj   = vj32
            #vj = jnp.einsum('ijkl,ji->kl', E, density_matrix)
            #vj = (E.reshape(d**2, d**2) @ density_matrix.reshape(-1)).reshape(d, d)
            #print("before") # looks like this fixes it, lol. 
            vj = jnp.sum(E.reshape(d**2, d**2) * density_matrix.reshape(1, -1), axis=1).reshape(d,d)
            #print("after")

            # point one; does the cond number thing help? 
            # 

            # so this trick gives one thing: compute difference with arbitrary cond number for electron repulson. 
            # we just need a 10x improvement on vk
            #d = density_matrix.shape[0]
            #vj   = (E.reshape(d**2, d**2)*args.scale_vj @ density_matrix.T.reshape(d**2)).reshape(d,d)/args.scale_eri            
            #vj_I = (( E.reshape(d**2, d**2)*c + I*c ) @ density_matrix.reshape(d**2)).reshape(d,d)
            #vj   = vj_I - density_matrix.reshape(d, d) *c
            #vj = vj/c # we could even store vj for loss in c format and scale down in end. 

            d = switch 
            vj, density_matrix = f(vj, d), f(density_matrix, d) 

            
            d = 12 in args.sk  
            density_matrix, E = f(density_matrix, d), f(electron_repulsion, d), 
            #vk = jnp.einsum('ijkl,jk->il', E, density_matrix)*jnp.asarray(hyb , dtype=E.dtype) #/ args.scale_eri 
            d = density_matrix.shape[0]

            def asort(x):
                print(x.shape)
                #x = x[:, jnp.argsort(jnp.abs(x), axis=0)]
                #x = jnp.take_along_axis(x, jnp.argsort(jnp.abs(x), axis=1), axis=1)
                print(x.shape)
                return jnp.sort(x, axis=1) # sorting is too slow https://github.com/google/jax/issues/10434
            vk = jnp.sum(E.transpose(1,2,0,3).reshape(d**2, d**2) * density_matrix.reshape(1, -1), axis=1).reshape(d,d)*jnp.asarray(hyb, dtype=E.dtype)
            #vk = jnp.sum(asort(E.transpose(1,2,0,3).reshape(d**2, d**2) * density_matrix.reshape(1, -1)), axis=1).reshape(d,d)*jnp.asarray(hyb, dtype=E.dtype)
            #vk = kahan_matvec(E.transpose(1,2,0,3).reshape(d**2, d**2) * density_matrix.reshape(1, -1)).reshape(d, d) *jnp.asarray(hyb, dtype=E.dtype)
            d = switch 
            vk, density_matrix = f(vk, d), f(density_matrix, d)

            # might even do one iteration on cpu in float32? might just even be the eri step?  one CPU eri mult takes 0.043ms
            # will be too slow when scaling I think? that said, could use entire CPU for this, so basically as long as our DFT isn't 50x faster it's ok? 
           
    else:
        Z = jnp.empty(density_matrix.shape)
        vj, vk = Z, Z 

    # this is vk and vj stuff 
    d = 13 in args.sk  
    V_xc, vj, vk = f(V_xc, d), f(vj, d), f(vk, d)
    #V_xc      += vj - vk * .5                                           
    #V_xc      = V_xc + vj/args.scale_vj/args.scale_eri - vk/args.scale_eri * .5
    #V_xc      = (V_xc*args.scale_eri + vj/args.scale_vj- vk* .5)/args.scale_eri
    #vj_m_vk = vj_I - vk_I*hyb/2
    vj_m_vk = vj/c - vk/2
    V_xc      = (V_xc+ vj_m_vk )
    #V_xc      = (V_xc + vj32/args.scale_vj) - vk * .5
    #V_xc      = (V_xc + vj32/args.scale_vj) - vk * .5 # perhaps we subtract vk and add it afterwards? may have a fun physical interpretation 
    d = switch 
    V_xc, vj, vk = f(V_xc, d), f(vj, d), f(vk, d)


    d = 14 in args.sk  # 73.103719
    density_matrix, E_xc, vk = f(density_matrix, d), f(E_xc, d), f(vk, d)
    #E_xc      -= jnp.sum(density_matrix * vk.T) * .5 * .5 
    #E_xc      = E_xc - jnp.sum(density_matrix * vk.T) *( .5 * .5) 
    if args.float32 and not args.backend == "ipu": E_xc      = E_xc - kahan_dot(density_matrix.reshape(-1) , vk.T.reshape(-1)) *( .5 * .5) 
    else: E_xc      -= jnp.sum(density_matrix * vk.T) * .5 * .5 # vj isn't used here! 
    #E_xc      = E_xc - kahan_dot(density_matrix.reshape(-1) , vk.T.reshape(-1)) *( .5 * .5) 
    d = switch 
    density_matrix, E_xc, vk = f(density_matrix, d), f(E_xc, d), f(vk, d)

    #d = 10 in args.sk #  56.137293
    #density_matrix, vj = f(density_matrix, d) , f(vj, d)
    #mult = (density_matrix * vj.T).reshape(-1)
    #d = False 
    #mult, density_matrix, vj = f(mult, d), f(density_matrix, d) , f(vj, d)

    # this is vj stuff 
    #d = 15 in args.sk # 12.915521
    #mult = f(mult, d) # this was perhaps the most tricky and we don't even need to do it on device; can do in end. 
    E_coulomb  = 0 # jnp.sum(density_matrix * vj.T) * .5 / args.scale_vj   # interestnig; i imagine this wouldn't give that large errors 
    #d = False 
    #E_coulomb, mult = f(E_coulomb, d), f(mult, d)


    return E_xc, V_xc, E_coulomb, vj, vk

       
def _eigh(x): 
    if args.backend == "ipu": 
        t0 = time.time()
        print("tracing ipu eigh (%s): "%str(x.shape))
        from jax_ipu_research.tile import ipu_eigh
        n = x.shape[0]
        pad = n % 2
        print(x.dtype)
        if pad:
            x = jnp.pad(x, [(0, 1), (0, 1)], mode='constant')

        eigvects, eigvals = ipu_eigh(x, sort_eigenvalues=True, num_iters=12)    # would more iteration make more accurate? 

        if pad: 
            e1 = eigvects[-1:]
            col = jnp.argmax(e1)
            eigvects = jnp.roll(eigvects, -col-1)
            eigvects = eigvects[:, :-1] 
            eigvects = jnp.roll(eigvects, -(-col))
            eigvects = eigvects[:-1]
    else: 
        eigvals, eigvects = jnp.linalg.eigh(f(x, args.float32))  # control this in float64? 
        
    return eigvals, eigvects 

def DIIS(cycle, sdf, hamiltonian, _V, _H, mf_diis_H): 
    # Update hamiltonian as linear combination of previous iterations 
    mf_diis_head      = cycle % _V.shape[0] 
    nd, d             = _V.shape            

    errvec = (sdf - sdf.T) 

    # Store current (hamiltonian,errvec) as flattened as row inside _V and _H.
    _V = jax.lax.dynamic_update_slice(_V, errvec.reshape(1, d),      (mf_diis_head, 0))  
    _H = jax.lax.dynamic_update_slice(_H, hamiltonian.reshape(1, d), (mf_diis_head, 0))  

    tmps = (_V.reshape(nd, 1, d) @ errvec.reshape(1, d, 1))
    tmps = tmps.reshape(-1)                                                             

    # Shapes in initial code depended on min(cycle, _V.shape[0]). 
    # To allow jax.jit, we always use nd=_V.shape[0] and zero out 
    # the additional stuff with the following mask.
    mask = jnp.where(np.arange(_V.shape[0]) < jnp.minimum(cycle+1, _V.shape[0]),         # min(cycle, _V) => min(cycle+1, _V) fixed it! 
                        jnp.ones(_V.shape[0], dtype=_V.dtype), jnp.zeros(_V.shape[0], dtype=_V.dtype))
    tmps = tmps * mask                                            

    # Assign tmp into row/col 'mf_diis_head+1' of mf_diis_H 
    mf_diis_H = jax.lax.dynamic_update_slice( mf_diis_H, tmps.reshape(1, -1), (mf_diis_head+1, 1) )
    mf_diis_H = jax.lax.dynamic_update_slice( mf_diis_H, tmps.reshape(-1, 1), (1, mf_diis_head+1) )

    # Compute new hamiltonian as linear combination of previous 8.
    # Coefficients are computed as pseudo_inverse of mf_diis_H. 
    # The first 8 iterations we are constructing mf_diis_H so it has shape (2,2), (3,3), (4,4), ...
    # To allow jax.jit we pad to (9, 9) and just zero out the additional stuff... 
    mask_            = jnp.concatenate([jnp.ones(1, dtype=mask.dtype), mask])                                    # the mask here defaulted to float64.
    masked_mf_diis_H = mf_diis_H[:nd+1, :nd+1] * mask_.reshape(-1, 1) * mask_.reshape(1, -1)     

    if args.backend == "ipu":  # this could be it? 
        #c               = pinv( masked_mf_diis_H )[0, :] 
        #print(masked_mf_diis_H.dtype)
        c               = pinv0( masked_mf_diis_H )
        #c               = jnp.linalg.pinv( masked_mf_diis_H )[0, :]  # this uses eigh, so possible pinv is better choice? 
    else: 
        c = jnp.linalg.pinv(f(masked_mf_diis_H, args.float32))[0, :] # jnp here eigh is inaccurate; cause for problem here aswell? 
        #c = pinv(f(masked_mf_diis_H, args.float32))[0, :]
        #c = pinv0(f(masked_mf_diis_H, args.float32))

    
    scaled_H         = _H[:nd] * c[1:].reshape(nd, 1)                         
    hamiltonian      = jnp.sum( scaled_H, axis=0 ).reshape(hamiltonian.shape)               

    return hamiltonian, _V, _H, mf_diis_H, errvec
    

def pinv(a):  # take out first row
    #cond = 10. * 9 * 1.1920929e-07 
    cond =  9*1.1920929e-07 
    #cond =  1.1920929e-07 
    vals, vect = _eigh ( a )
    return (vect @ jnp.diag(jnp.where( jnp.abs(vals) > cond, 1/vals, 0)) @ vect.T)#[0, :] rewrite to entry-wise mult? less numerical error also if we multiply directly on?!

def pinv0(a):  # take out first row
    #cond = 10. * 9 * 1.1920929e-07 
    cond =  9*1.1920929e-07 
    #cond =  1.1920929e-07 
    vals, vect = _eigh ( a )
    #c = (vect @ jnp.diag(jnp.where( jnp.abs(vals) > cond, 1/vals, 0)) @ vect.T)[0, :] #rewrite to entry-wise mult? less numerical error also if we multiply directly on?!
    #print(c)
    #e1 = np.zeros(a.shape[0])
    #e1[0] = 1
    #e1 = jnp.asarray(e1)
    #c = vect @ (jnp.diag(jnp.where( jnp.abs(vals) > cond, 1/vals, 0)) @ (vect.T @ e1)) #rewrite to entry-wise mult? less numerical error also if we multiply directly on?!
    #c = vect @ (jnp.diag(jnp.where( jnp.abs(vals) > cond, 1/vals, 0)) @ (vect[0, :])) #rewrite to entry-wise mult? less numerical error also if we multiply directly on?!
    c = vect @ ( jnp.where( jnp.abs(vals) > cond, 1/vals, 0) * vect[0, :]) #rewrite to entry-wise mult? less numerical error also if we multiply directly on?!
    return c

table = None
def recompute(args, molecules, id, num, our_fun, str="", atom_string=""):
  global table 
  t0 = time.time()


  if str == "": 
    atoms = molecules["atom_string"][id]
    locs  = molecules["atom_locations"][id]*angstrom_to_bohr

    atom_string, str = get_atom_string(atoms, locs)


  def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

  mol = Mole()
  #args.atoms = [a for a in str.replace(";", "").split(" ") if not is_float(a) and a != ""] 
  #args.locs  = np.array([float(a) for a in str.replace(";", "").split(" ") if is_float(a)]).reshape(-1, 3)

  mol.build(atom=str, unit="Bohr", basis=args.basis, spin=args.spin, verbose=0)

  print("\t", atom_string, end="")

  if not args.skipus:  
    energies, our_energy, our_hlgap, t_us, t_main_loop, us_hlgap = our_fun(str)

  if not args.skippyscf: 
    mol = Mole(atom=str, unit='Bohr', basis=args.basis, spin=args.spin) 
    mol.verbose = 0 
    __config__.dft_rks_RKS_grids_level = args.plevel 
        
    mol.max_cycle = args.its 
    mf = scf.RKS(mol) 
    mf.xc = args.xc 

    if args.skipdiis: 
      print("\n[ TURNING OFF DIIS IN PYSCF ]")
      mf.diis = 0 
      mf.diis_space = 0 

    mf.diis_space = 9 # checked worked by setting to 0 which breaks bookeep pyscf uses 

    repeats = 1
    if args.benchmark: repeats = 3

    if args.gdb == 13 and False:
        if args.level == 0: pyscf_energy = -12594.313950 

    elif args.gdb == 11 and False: 
        if args.level == 3: pyscf_energy = -15000.689408
        if args.level == 2: pyscf_energy = -15000.689408  # grid-size 2? ;; try 5 conformers for each molecule, take the most stable one? 
        if args.level == 1: pyscf_energy = -15000.685109 
        if args.level == 0: pyscf_energy = -15000.428090
    else: 

        pyscf_energies = []
        pyscf_dms = []
        def callback(envs):
            energy = envs['e_tot'] #
            pyscf_energies.append(energy)
            pyscf_dms.append(envs["dm"])

        mf.callback = callback


        print("[ pyscf ] ", end="")
        for _ in range(repeats):
            t0 = time.time()
            pyscf_energy    = mf.kernel()  * hartree_to_eV
            t_pyscf = time.time()-t0
            print(" %4fs"%(t_pyscf), end="")
            print("")

        print("PYSCF iterations: ", len(pyscf_energies))
        print(np.array(pyscf_energies)*hartree_to_eV)

        lumo           = np.argmin(mf.mo_occ)
        homo           = lumo - 1
        hl_gap_hartree = np.abs(mf.mo_energy[homo] - mf.mo_energy[lumo])

        print("pyscf_hlgap\t%15f"%( hl_gap_hartree * hartree_to_eV))
        print("us_hlgap\t%15f"%(    us_hlgap))
        print("err_hlgap\t%15f"%np.abs((hl_gap_hartree * hartree_to_eV) - us_hlgap))

  else: 
    pyscf_energy, pyscf_hlgap, t_pyscf = -1, -1, -1 # np.load("tmp/%i_%s"%(args.num, atom_string)) # do like this? 

  if molecules is not None: pcq_hlgap = molecules["hlgap"][id]
  else: pcq_hlgap = -1

  print("pyscf:\t\t%15f"%pyscf_energy)
  print("us:\t\t%15f"%our_energy)
  print("mus:\t\t%15f"%np.mean(energies[-10:]))
  print("diff:\t\t%15f"%np.abs(pyscf_energy-our_energy))
  # plot the result of hte numerical errors; does it looks like a gaussian? 
  print("mdiff:\t\t%15f"%np.abs(pyscf_energy-np.mean(energies[-10:])), np.std(energies[-10:])) # std / \sqrt(n) => if they were normally distributed 10
  #print("mdiff:\t\t%15f"%np.abs(pyscf_energy-np.mean(energies[-50:])), np.std(energies[-50:]))
  print("chemAcc: \t%15f"%0.043)
  print("chemAcc/diff: \t%15f"%(0.043/np.abs(pyscf_energy-our_energy)))
  print("chemAcc/mdiff: \t%15f"%(0.043/np.abs(pyscf_energy-np.mean(energies[-10:]))))
  print("> diffs:")
  print(np.abs(energies.reshape(-1)[-5:] - pyscf_energy))
  print("mean+-var: %f +- %f"%( np.mean(np.abs(energies.reshape(-1)[-5:] - pyscf_energy)), np.var(np.abs(energies.reshape(-1)[-5:] - pyscf_energy))))
  fig, ax = plt.subplots(1,2)
  ax[0].plot(np.abs(energies)[-10:], label="JaxDFT")
  ax[0].plot(np.arange(10), np.ones(10)*np.mean(np.abs(energies)[-10:]), label="Mean")
  ax[0].plot(np.arange(10), np.ones(10)*np.abs(pyscf_energy), label="target")
  ax[1].hist(energies[-200:], bins=20)
  ax[1].plot([pyscf_energy, pyscf_energy], [0, 10], label="truth")
  plt.legend()
  plt.yscale("log")
  plt.savefig("numerror.jpg")
  #134
  #136
  #137
  #138
  #140
  #142
  #144
  #146
  #148
  #150
  #152
  #154
  #158
  #160
  #162
  #165
  #170
  #172
  #175
  # select molecules based on variance in last 10 iterations; 
  # select by less than something, if not, discard. 
  # looks like error correlates with variance in last 10 iterations. 

  # do normal PySCF reference DFT on the entire validation set. 
  # benchmark against that. 


def test_cases(args, molecules, our_fun=lambda x: (0., 0., 0.), stop=-1, list=None): 

  ids = molecules["sdf_id"].iloc

  if list is not None: 
    for num, i in enumerate(list[args.skip::args.step]): 
      #if num < args.skip: continue
      if num in [10, 72]: continue # don't do spin=1 ones!  
      id = ids[i]
      recompute(args, molecules, id, num, our_fun= our_fun)

  else: 
    for num, id in enumerate(ids[args.skip::args.step]): 
      #if num < args.skip: continue
      if num == stop: break
      if num in [10, 72]: continue # don't do spin=1 ones! 
      recompute(args, molecules, id, num, our_fun= our_fun)


mol = None 
_str = None 
def jax_dft(str):
    global mol 
    # So PCQ uses eV and PySCF uses Hartree 

    mol = Mole()
    _str = str
    mol.build(atom=str, unit="Bohr", basis=args.basis, spin=args.spin, verbose=0)

    n_electrons       = mol.nelectron
    n_electrons_half  = n_electrons//2
    N                 = mol.nao_nr()

    if args.num == -1 or args.benchmark:  # Don't print this if running many of the molecules e.g. num = 10 
        print("")
        print("> benchmarking ")
        print("[ basis set] ", args.basis)
        print("[ num_ao   ] ", mol.nao_nr())
        print("[ eri MB   ] ", mol.nao_nr()**4*4/10**6, (mol.nao_nr()**2, mol.nao_nr()**2), " !!not sparsified / symmetrized!!")
        print("[ atom_str ] ", str) 

    repeats = 1
    if args.benchmark: repeats = 6

    atom_positions  = jnp.concatenate([np.array(atom_position).reshape(1,3) for atom_symbol, atom_position in mol._atom], axis=0) 

    ts = []
    for step in range(repeats):
        t0 = time.time()
        vals = density_functional_theory(atom_positions) 
        energies, e_tot, mo_energy, mo_coeff, dms, L_inv = [np.asarray(a) for a in vals]
        #jax.block_until_ready(energies) 
        t = time.time() - t0
        e_tot = energies[-1]*hartree_to_eV

        print(energies*hartree_to_eV)

        if args.benchmark: print("[ it %4f ]"%t)

    mo_occ = jnp.concatenate([jnp.ones(n_electrons_half)*2, jnp.zeros(N-n_electrons_half)])

    # mean doesn't seem to help much here?
    #mo_energy = np.linalg.eigh(L_inv @ np.mean(dms[-10:, 2], axis=0).reshape(53,53) @ L_inv.T)[0]
    d = L_inv.shape[0]
    mo_energy = np.linalg.eigh(L_inv @ np.mean(dms[-1:, 2], axis=0).reshape(d,d) @ L_inv.T)[0]
    lumo            = np.argmin(np.array(mo_occ))
    homo            = lumo - 1
    hl_gap_hartree  = np.abs( mo_energy[homo] - mo_energy[lumo] )
    hlgap           = hartree_to_eV*hl_gap_hartree

    return energies * hartree_to_eV, e_tot, hlgap, t, t, hlgap


if __name__ == "__main__":  # need to do a bit of work not to OOM. 
    parser = argparse.ArgumentParser(description='Arguments for Density Functional Theory. ')
    parser.add_argument('-num_conformers', default=1000, type=int, help='How many rdkit conformers to perfor DFT for. ')
    parser.add_argument('-nohs', action="store_true", help='Whether to not AddHs (default add Hs). ')
    parser.add_argument('-limit', default=-1, type=int, help='smiles = args.smiles[:limit]; useful for never doing too many hydrogens. ')
    parser.add_argument('-verbose', action="store_true")

    parser.add_argument('-choleskycpu',       action="store_true", help='Whether to do np.linalg.inv(np.linalg.cholesky(.)) on cpu/np/f64. ')
    parser.add_argument('-resume',       action="store_true", help='Whether to do not do molecules already found in storage folder. ')

    parser.add_argument('-density_mixing',       action="store_true", help='Compute dm=(dm_old+dm)/2')
    parser.add_argument('-skip_minao',       action="store_true", help='In generation mode re-uses minao init across conformers.')

    parser.add_argument('-num',       default=10,          type=int,   help='Run the first "num" test molecules. ')
    parser.add_argument('-id',        default=126,          type=int,   help='Run only test molecule "id". ')
    parser.add_argument('-its',       default=50,          type=int,   help='Number of Kohn-Sham iterations. ')
    parser.add_argument('-step',      default=1,           type=int,   help='If running 1000s of test cases, do 1, 1+step, 1+2*step, ... ')
    parser.add_argument('-spin',      default=0,           type=int,   help='Even or odd number of electrons? Only tested for spin=0')
    parser.add_argument('-str',       default="",          help='Molecule string, e.g., "H 0 0 0; H 0 0 1; O 1 0 0; "')
    parser.add_argument('-init',      default="minao",     help='How to initialize density matrix. ')

    parser.add_argument('-numerror', action="store_true",     help='Save all tensors to wandb. ')

    parser.add_argument('-ipumult', action="store_true",     help='On ipu do mult using normal eri. ')


    parser.add_argument('-skippyscf', action="store_true", help='Skip PySCF. ')
    parser.add_argument('-skipus',    action="store_true", help='Skip our code. ')

    parser.add_argument('-float32',   action="store_true", help='Whether to use float32 (default is float64). ')
    parser.add_argument('-float16',   action="store_true", help='Whether to use float16 (default is float64). ')
    parser.add_argument('-basis',     default="STO-3G",    help='Developed for STO-3G (and a little for 6-31G*), but others may work. ')
    parser.add_argument('-xc',        default="b3lyp",     help='Only implemented b3lyp (and it\'s components); known bug exists for lda. ')
    parser.add_argument('-loadpyscf', action="store_true", help='Load precomputed PySCF values, a little faster. ') 
    parser.add_argument('-benzene',   action="store_true", help='Benchmark time on benzene, see https://pyscf.org/benchmark.html. ') 
    parser.add_argument('-skip',      default=0,           help='Skip the first "skip" testcases. ', type=int) 
    parser.add_argument('-backend',   default="cpu",       help='Which backend to use, accepts {cpu, ipu, gpu} dependent on hardware. ') 
    parser.add_argument('-wandb',     action="store_true", help='Whether to store results in wandb.') 

    parser.add_argument('-benchmark', action="store_true", help='Print benchmark info inside our DFT computation. ') 
    parser.add_argument('-nan',       action="store_true", help='Whether to throw assertion on operation causing NaN, useful for debugging NaN arrising from jax.grad. ')
    parser.add_argument('-skipdiis',  action="store_true", help='Whether to skip DIIS; useful for benchmarking gradient code. ')
    parser.add_argument('-skipeigh',  action="store_true", help='Whether to skip eigh; useful for benchmarking gradient code. ')

    parser.add_argument('-methane',  action="store_true", help='simulate methane . ')
    parser.add_argument('-H',        action="store_true", help='simple hydrogen system. ')

    parser.add_argument('-forloop',  action="store_true", help="Runs SCF iterations in python for loop; allows debugging on CPU, don't run this on IPU super slow. ")

    parser.add_argument('-water',    default=-1, nargs='+', help="Run simulation on water with random positions. ", type=int)
    parser.add_argument('-he',       action="store_true", help="Just do a single He atom, fastest possible case. ")

    parser.add_argument('-level',    default=2, help="Level of the grids used by us (default=2). ", type=int)
    parser.add_argument('-plevel',   default=2, help="Level of the grids used by pyscf (default=2). ", type=int)

    parser.add_argument('-C',         default=-1, type=int,  help='Number of carbons from C20 . ')
    parser.add_argument('-biochem',        default=-1, type=int,  help='Number of atoms from the important biochemistry ones {C,H,N,O,F,S}. ')
    parser.add_argument('-gdb',        default=-1, type=int,  help='Which version of GDP to load {10, 11, 13, 17}. ')
    parser.add_argument('-gdb13',        default=-1, type=int,  help='Which split of GDB 13 to do (we split gdb13 into 100 versions). ')

    parser.add_argument('-skiperi',         action="store_true", help='estimate time if eri wasn"t in computation. done by using (N, N) matrix instead. ')
    parser.add_argument('-randeri',         action="store_true", help='Initialize electron_repulsion=np.random.normal(0,1,(N,N,N,N))')

    parser.add_argument('-generate',         action="store_true", help='Generate dataset and store in wandb. ')
    parser.add_argument('-save',         action="store_true", help='Save generated data to wandb! ')

    parser.add_argument('-fname',    default="", type=str, help='Save stuff in this folder name. ')

    parser.add_argument('-multv',    default=2, type=int, help='mult version. different versions to to multiply ERI@flat(v); has different trades-offs for memory vs speed. ')
    parser.add_argument('-intv',    default=1, type=int, help='integral version. different versions to compute integrals. ')

    parser.add_argument('-scale_eri',       default=1, type=float,  help='Scale electron repulsion ')

    parser.add_argument('-scale_w',         default=1, type=float,  help='Scaling of weights to get numerical stability. ')
    parser.add_argument('-scale_ao',        default=1, type=float,  help='Scaling of ao to get numerical stability. ')
    parser.add_argument('-scale_overlap',   default=1, type=float,  help='Scaling of overlap to get numerical stability. ')

    parser.add_argument('-scale_cholesky',  default=1, type=float,  help='Scale electron repulsion ')
    parser.add_argument('-scale_ghamil',  default=1, type=float,  help='Scale electron repulsion ')
    parser.add_argument('-scale_eigvects',  default=1, type=float,  help='Scale electron repulsion ')

    parser.add_argument('-scale_sdf',  default=1, type=float,  help='Scale electron repulsion ')

    parser.add_argument('-scale_vj',  default=1, type=float,  help='Scale electron repulsion ')

    parser.add_argument('-scale_errvec',  default=1, type=float,  help='Scale electron repulsion ')

    parser.add_argument('-sk',  default=[-2], type=int, nargs="+", help='Scale electron repulsion ')

    parser.add_argument('-jit',  action="store_true")
    parser.add_argument('-enable64',  action="store_true")

    parser.add_argument('-rattled_std',  type=float, default=0, help="Add N(0, args.ratled_std) noise to atom positions before computing dft. ")

    parser.add_argument('-profile',  action="store_true")
    parser.add_argument('-pyscf',  action="store_true", help="Used to compute with reference implementation. Example: Compute all of QM9 with PySCF to test our code. ")
    parser.add_argument('-uniform_pyscf',  default = -1, type=float, help="Do reference implementation if 'np.random.uniform(0,1,(1))<args.uniform_pyscf'") 

    parser.add_argument('-threads',  default=1, type=int, help="Number of threads to use to compute ipu_mult_direct. ")
    parser.add_argument('-threads_int',  default=1, type=int, help="Number of threads to use to do int2e_sph. ")
    parser.add_argument('-split',  default=[1, 16], type=int, nargs="+", help='How to split between servers; e.g., [1, 16] means this is IPU num 1 out of 16. ')

    parser.add_argument('-debug',  action="store_true", help='Used to turn on/off the f function used to investigate float{32,64}. ')

    parser.add_argument('-seperate',  action="store_true", help='Used to seperate electron integral computation from DFT computation to lower memory consumption. ')

    parser.add_argument('-gname',  default="", type=str, help='Folder name to store generate dataset; useful when using multiple pods to generate. ')

    parser.add_argument('-checkc',  action="store_true" , help='Check convergence; plot energy every iteration for us and pyscf. ')
    
    parser.add_argument('-geneigh',  action="store_true" , help='Use generalized eigendecomposition like pyscf; relies on scipy, only works in debug mode with -forloop. ')

    args = parser.parse_args()



    print(sys.argv)

    print(natsorted(vars(args).items()) )

    print("[BASIS]", args.basis)

    if -1 in args.sk: args.sk = list(range(20))
    print(args.sk)

    if args.checkc: 
        args.pyscf = True

    if args.pyscf:
        args.verbose = True 
    
    if args.backend == "cpu":
        args.seperate = False 


    if True:  # after we got caching, we can very quickly do HPO on scaling parameters; holy shit that's cool! 
        # but we'll have to rewrite scaling as parameters that goes into the graph, not get's compiled down. 
        args.scale_w = 1
        args.scale_ao = 1
        if args.float32: args.scale_eri = 1
        args.scale_eri = 1  #128 # perhaps we need to tune this? 
        args.scale_cholesky= 1
        args.scale_ghamil = 1
        args.scale_eigvects = 1


    sys.argv = sys.argv[:1] 

    if args.numerror: 
        args.forloop = True 
        args.backend = "cpu"
        args.its = 35 

    print("")

    if args.backend == "ipu":  # allowing use of cpu float64 in jnp while using float32 on ipu 
        args.float32 = True 
        #args.sk = np.arange[-1 
        args.sk = list(range(20))
        args.debug = True 

    if args.float32 or args.float16: 
        #EPSILON_B3LYP  = 1e-30
        #CLIP_RHO_MIN   = 1e-8
        #CLIP_RHO_MAX   = 1e30

        if args.enable64: config.update('jax_enable_x64', True) # on/off gives ~ the same! 
        EPSILON_B3LYP  = 1e-20
        CLIP_RHO_MIN   = 1e-9
        CLIP_RHO_MAX   = 1e12

    else:  # float64
        config.update('jax_enable_x64', True) # perhaps it's the loss computation in the end? 
        EPSILON_B3LYP  = 1e-20 
        #CLIP_RHO_MIN   = 1e-10 
        CLIP_RHO_MIN   = 1e-9 
        CLIP_RHO_MAX   = 1e12

    if args.nan: 
        config.update("jax_debug_nans", True)

    backend = args.backend
    eigh = _eigh

    molecules = pd.read_pickle("data/unique.pkl")

    if args.str != "": 
        recompute(args, None, 0, 0, our_fun=jax_dft, str=args.str)

    elif args.gdb > 0:       

        # one strategy could just be to split gdb into a lot of chunks, and solve each chunk independently? 

        if args.gdb != 20 or True:     
            t0 = time.time()
            print("loading gdb data")
            if args.gdb == 13: 
                t0 = time.time()
                #args.smiles = pd.read_csv("gdb/13.cno.sorted.49M.csv", nrows=1000000).values[100:][:, 1]
                #args.smiles = pd.read_csv("gdb/13.cno.sorted.1000000.csv").values[:, 1]
                args.smiles = pd.read_parquet("gdb/gdb13/00_v3.parquet")["smiles"].values
                print(args.smiles)
                print(time.time()-t0)
                #print(args.smiles[3])
                #exit()
            if args.gdb == 11: 
                #args.smiles = [a.split(" ")[0] for a in open("gdb/gdb11_size11.smi", "r").read().split("\n")]
                args.smiles = [a for a in open("gdb/gdb11_size11_sorted.csv", "r").read().split("\n")]
                print(len(args.smiles))

            if args.gdb == 12: 
                t0 = time.time()
                args.smiles = pd.read_parquet("gdb/gdb12/gdb12_sorted.parquet")["smiles"].values # takes ~ 30s to load? 
                #print(time.time()-t0)
                #print(args.smiles)

            if args.gdb == 10: args.smiles = [a for a in open("gdb/gdb11_size10_sorted.csv", "r").read().split("\n")]
            if args.gdb == 9:  args.smiles = [a for a in open("gdb/gdb11_size09_sorted.csv", "r").read().split("\n")]
            if args.gdb == 7:  args.smiles = [a for a in open("gdb/gdb11_size07_sorted.csv", "r").read().split("\n")]
            if args.gdb == 8:  args.smiles = [a for a in open("gdb/gdb11_size08_sorted.csv", "r").read().split("\n")]
            if args.gdb == 6:  args.smiles = ["c1ccccc1"]*1000 # do benzene for timing experiment. 
            if args.gdb == 5:  args.smiles = ['CCCCC']*1000
            if args.gdb == 4:  args.smiles = ['CCCC']*1000

            if args.gdb == 1337: 
                args.qm9 = pd.read_pickle("../petrignn/schnet/pyg_qm9.pickle")
                args.smiles = ["CCCCCCCCCC"]*args.qm9.shape[0] # dummy smile string to fix grid size. 

            print("DONE!", time.time()-t0)

            print("Length GDB: ", len(args.smiles))

            if args.limit != -1:
                args.smiles = args.smiles[:args.limit]
            
            for i in range(int(args.id), int(args.id)+1000):
                smile = args.smiles[i] 
                smile = smile

                print(smile)

                #atoms = [a for a in list(smile) if a == "C" or a == "N" or a == "O" or a == "F"] #+ ["H"]
                b = Chem.MolFromSmiles(smile)
                #b = Chem.AddHs(b)  # perhaps more numerically stable wiht the H's? e.g. energy may be lower?;; main problem with this is memory? 
                if not args.nohs: b = Chem.AddHs(b, explicitOnly=False)   # AddExplici,tHs AddPolarHss
                atoms = [atom.GetSymbol() for atom in b.GetAtoms()]

                # oh wait, so the problem is not that the isomers are bad, it's that the smiles stirng we read in are bad? :OOOO
                e = AllChem.EmbedMolecule(b) # hypothesis: adding Hs on average increase energy => increase numerror, but, removes extreme cases? 
                if e == -1: continue
                # does look like that!  main problem of more Hs is increased memory consumption, so if we fix that it looks pretty good! 

                #locs = np.concatenate( (b.GetConformer().GetPositions(), np.ones((1,3))*4), axis=0)
                locs = b.GetConformer().GetPositions() * angstrom_to_bohr
                atom_string, string = get_atom_string(" ".join(atoms), locs)

                print(string)
                break
            #exit()
            recompute(args, None, 0, 0, our_fun=jax_dft, str=string) # how do 

    elif args.gdb13 >= 0: 
        t0 = time.time()
        #df = pd.read_parquet("gdb/gdb13/%02i_v2_sorted.parquet"%int(args.gdb13))
        df = pd.read_parquet("gdb/gdb13/%02i_v3.parquet"%int(args.gdb13)) # load the sorted gdb 

        df = df.sort_values("hs")[1:] # there's only one thing with one hydrogen!  # try to just skip odd N? 

        print("Loaded 'gdb/gdb13/%02i_sorted.parquet' in %5fs. "%(int(args.gdb13), time.time()-t0))
        print(df)

        args.smiles = df["smiles"].values

        print("DONE!", time.time()-t0)

        print("Length GDB: ", len(args.smiles))
        
        smile = args.smiles[int(args.id)] 
        smile = smile

        for smile in tqdm(args.smiles): 
            #atoms = [a for a in list(smile) if a == "C" or a == "N" or a == "O" or a == "F"] #+ ["H"]
            b = Chem.MolFromSmiles(smile)
            #b = Chem.AddHs(b)  # still is super slow for some reason? 
            if not args.nohs: b = Chem.AddHs(b, explicitOnly=False)   # AddExplici,tHs AddPolarHss
            atoms = [atom.GetSymbol() for atom in b.GetAtoms()]

            # we could also try to generate ~ 50 isomers for the GDB12 dataset? (this one has how many?)
            emb = AllChem.EmbedMolecule(b, maxAttempts=1) # hypothesis: adding Hs on average increase energy => increase numerror, but, removes extreme cases? 
            if emb != -1: break 
            #print(string)

        print(smile)

        locs = b.GetConformer().GetPositions() * angstrom_to_bohr
        atom_string, string = get_atom_string(" ".join(atoms), locs)

        print(string)
        #exit()
        recompute(args, None, 0, 0, our_fun=jax_dft, str=string) # how do 
        
        #exit()
        recompute(args, None, 0, 0, our_fun=jax_dft, str=string) # how do
    

    elif args.C > 0:       
        _str = ";".join("C     1.56910  -0.65660  -0.93640;\
        C     1.76690   0.64310  -0.47200;\
        C     0.47050  -0.66520  -1.79270;\
        C     0.01160   0.64780  -1.82550;\
        C     0.79300   1.46730  -1.02840;\
        C    -0.48740  -1.48180  -1.21570;\
        C    -1.56350  -0.65720  -0.89520;\
        C    -1.26940   0.64900  -1.27670;\
        C    -0.00230  -1.96180  -0.00720;\
        C    -0.76980  -1.45320   1.03590;\
        C    -1.75760  -0.63800   0.47420;\
        C     1.28780  -1.45030   0.16290;\
        C     1.28960  -0.65950   1.30470;\
        C     0.01150  -0.64600   1.85330;\
        C     1.58300   0.64540   0.89840;\
        C     0.48480   1.43830   1.19370;\
        C    -0.50320   0.64690   1.77530;\
        C    -1.60620   0.67150   0.92310;\
        C    -1.29590   1.48910  -0.16550;\
        C    -0.01020   1.97270  -0.00630;".split(";")[:args.C])

        print(_str)
        recompute(args, None, 0, 0, our_fun=jax_dft, str=_str)

    elif args.biochem > 0:       
        _str = ";".join(
            "H     1.56910  -0.65660  -0.93640;\
             H     1.76690   0.64310  -0.47200;\
             C     0.47050  -0.66520  -1.79270;\
             N     0.01160   0.64780  -1.82550;\
             O     0.79300   1.46730  -1.02840;\
             F    -0.48740  -1.48180  -1.21570;\
             S    -1.56350  -0.65720  -0.89520;\
             C    -1.26940   0.64900  -1.27670;\
             N    -0.00230  -1.96180  -0.00720;\
             O    -0.76980  -1.45320   1.03590;\
             F    -1.75760  -0.63800   0.47420;\
             S     1.28780  -1.45030   0.16290;\
             C     1.28960  -0.65950   1.30470;\
             N     0.01150  -0.64600   1.85330;\
             O     1.58300   0.64540   0.89840;\
             F     0.48480   1.43830   1.19370;\
             S    -0.50320   0.64690   1.77530;".split(";")[:args.biochem])

        #print(_str)
        recompute(args, None, 0, 0, our_fun=jax_dft, str=_str)

    elif args.he: 
        recompute(args, None, 0, 0, our_fun=jax_dft, str="He 0 0 0; ")

    # change H to take a number, e.g., -H => H2 but -H 10 gives 10 H molecules. Do the same for water. 
    elif args.H:       recompute(args, None, 0, 0, our_fun=jax_dft, str="H 0 0 0; H 1 1 1;")
    elif args.methane: recompute(args, None, 0, 0, our_fun=jax_dft, str="C 0 0 0; H 0 0 1; H 1 0 0; H 0 1 0; H 1 1 0;")

    elif args.benzene: 
        benzene  = open("tmp/benzene.xyz", "r").read().split("\n")[2:] 
        
        def center(loaded_lst):
            # Center atom positions. TODO refactor this. 
            atoms = "".join([a.split(" ")[0] for a in loaded_lst])
            print(atoms)
            pos   = np.concatenate([np.array([float(a) for a in g.split()[1:]]).reshape(1,3) for g in loaded_lst])
            mean  = np.mean(pos, axis=0)
            pos   = pos - mean 

            # Get PySCF representation. 
            return get_atom_string(atoms, pos)[1]

        benzene  = center(benzene) 
        recompute(args, None, 0, 0, our_fun=jax_dft, str=benzene)
        assert False

        
    elif args.water != -1: 
        if 10 in args.water:
            atom_positions = np.load("1h2o.npz")["arr_0"]
            atom_positions = atom_positions - np.mean(atom_positions, axis=0).reshape(1, 3) # translate so mean is (0,0,0)

            water = ""
            for i in range(0, 10): 
                atom_O  = atom_positions[i*3]
                atom_H1 = atom_positions[i*3+1]
                atom_H2 = atom_positions[i*3+2]
                water += "O %f %f %f; H %f %f %f; H %f %f %f;"%(*atom_O.tolist(), *atom_H1.tolist(), *atom_H2.tolist())


            recompute(args, None, 0, 0, our_fun=jax_dft, str=water)

            assert False 


        for num in args.water: 
            water = ""
            np.random.seed(42)
            for i in range(num): 
                # O 1.779297 0.082581 -0.129883; H 3.607422 0.030426 -0.206665; H 1.257812 -0.904297 -1.580078;
                # no, make a cube 
                position = np.array([1.779297, 0.082581, -0.129883, 3.607422, 0.030426, -0.206665, 1.257812, -0.904297, -1.580078])  
                position[i%3::3] += 1 + i//3 # kid of makes a cube I think? 

                pos = tuple([position[j] for j in range(9)]) 
                water += "H %f %f %f; H %f %f %f; O %f %f %f;"%pos 

            recompute(args, None, 0, 0, our_fun=jax_dft, str=water)

    else: 
        test_cases(args, 
                our_fun    = jax_dft, 
                list       = range(int(args.num)) if args.id == -1 else [int(args.id)], 
                molecules  = molecules 
        )  

