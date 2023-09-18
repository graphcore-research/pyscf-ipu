# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import colored_traceback.always
import os
os.environ['OMP_NUM_THREADS'] = "8"
os.environ['TF_POPLAR_FLAGS'] = """--executable_cache_path=/tmp/pyscf-ipu-cache/"""
import jax
import jax.numpy as jnp
from jax.config import config
config.FLAGS.jax_platform_name = 'cpu'
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
from tqdm import tqdm
import os
import pyscf
from pyscf import scf
from pyscf import __config__
import argparse
from natsort import natsorted
import sys
import numpy as np
import pyscf.dft
from pyscf.scf import hf
from pyscf.gto.mole import Mole
from pyscf import scf
import time
import pandas as pd
import re
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from collections import namedtuple

from pyscf_ipu.pyscf_utils.minao            import minao
from pyscf_ipu.pyscf_utils.build_grid       import build_grid
from pyscf_ipu.pyscf_utils.build_mol        import build_mol
from pyscf_ipu.exchange_correlation.b3lyp   import b3lyp
from pyscf_ipu.exchange_correlation.b3lyp   import do_lda as lda
from pyscf_ipu.electron_repulsion.direct    import prepare_int_floats, prepare_integrals_2_inputs
from pyscf_ipu.electron_repulsion.direct    import (prepare_ipu_direct_mult_inputs,
                                                    prepare_integrals_2_inputs,
                                                    compute_integrals_2,
                                                    ipu_direct_mult)

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# Constants
angstrom_to_bohr = 1.88973
hartree_to_eV    = 27.2114

# Global variables, set from args
g_float32 = False
g_ipu = False
EPSILON_B3LYP  = 1e-20
CLIP_RHO_MIN   = 1e-9
CLIP_RHO_MAX   = 1e12


def get_atom_string(atoms, locs):
    atom_string = atoms
    atoms = re.findall('[a-zA-Z][^A-Z]*', atoms)
    str = ""
    for atom, loc in zip(atoms, locs):
      str += "%s %4f %4f %4f; "%((atom,) + tuple(loc) )
    return atom_string, str

_do_compute_static_argnums = (10,11,16)
def _do_compute(density_matrix, kinetic, nuclear, overlap, ao,
                electron_repulsion, weights, coords, 
                nuclear_energy, disable_cache, 
                mf_diis_space, N, hyb, mask, _input_floats,
                _input_ints, 
                args,
                L_inv=None):
        # --- INITIALIZE MATRICES USED FOR DIIS --- #
        mf_diis_H       = np.zeros((mf_diis_space+1, mf_diis_space+1))
        mf_diis_H[0,1:] = mf_diis_H[1:,0] = 1
        mf_diis_H       = np.array(mf_diis_H)

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

        indxs = None
        if args.backend == "ipu" and not args.ipumult:
            from pyscf_ipu.electron_repulsion.direct import prepare_integrals_2_inputs , compute_integrals_2
            _, _, _tuple_ijkl, _shapes, _sizes, _counts, indxs, indxs_inv, num_calls = prepare_integrals_2_inputs(mol)
            if not args.seperate:
                electron_repulsion = compute_integrals_2( _input_floats, _input_ints, _tuple_ijkl, _shapes, _sizes, _counts, tuple(indxs_inv), num_threads=args.threads_int, v=args.intv)[0]
                electron_repulsion = [a  for a in electron_repulsion]
            print("bytes: ", np.sum([a.nbytes for a in electron_repulsion])/ 10**6)
        elif not args.seperate:
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

        _num_calls = np.zeros(num_calls)
        cycle = 0
        if type(electron_repulsion) == type([]):
            if args.float32: E_xc, V_xc, E_coulomb, vj, vk = xc((args, indxs), density_matrix.astype(np.float32), dms.astype(np.float32), cycle, ao.astype(np.float32), electron_repulsion, weights.astype(np.float32), vj.astype(np.float32), vk.astype(np.float32), hyb, _num_calls)
        else:
            if args.float32: E_xc, V_xc, E_coulomb, vj, vk = xc((args, indxs), density_matrix.astype(np.float32), dms.astype(np.float32), cycle, ao.astype(np.float32), electron_repulsion.astype(np.float32), weights.astype(np.float32), vj.astype(np.float32), vk.astype(np.float32), hyb, _num_calls)
            else: E_xc, V_xc, E_coulomb, vj, vk = xc((args, indxs), density_matrix.astype(np.float64), dms.astype(np.float64), cycle, ao.astype(np.float64), electron_repulsion.astype(np.float64), weights.astype(np.float64), vj.astype(np.float64), vk.astype(np.float64), hyb, _num_calls)

        vals = [mask, allvals, cs, energies, V_xc, density_matrix, _V, _H, mf_diis_H,
                    vj, vk, eigvals, eigvects, energy, overlap, electron_repulsion,
                fixed_hamiltonian, L_inv, weights, hyb, ao, nuclear_energy, _num_calls, cholesky, generalized_hamiltonian, sdf, errvec, hamiltonian, dms, part_energies]

        vals = [f(a, args.float32) for a in vals]

        if args.nan or args.forloop:
            if args.jit: _iter = jax.jit(dft_iter, backend=args.backend)

            os.makedirs("numerror/%s/"%(str(args.sk)) , exist_ok=True)
            for n in tqdm(range(int(args.its))):

                if args.jit: vals = _iter((args,indxs), n, vals)
                else: vals = dft_iter((args,indxs), n, vals)

                if args.numerror:
                    _str = ["mask", "allvals", "cs", "energies", "V_xc", "density_matrix", "_V", "_H", "mf_diis_H",
                            "vj", "vk", "eigvals", "eigvects", "energy", "overlap", "electron_repulsion",
                            "fixed_hamiltonian", "L_inv", "weights", "hyb", "ao", "nuclear_energy", "np.zeros(num_calls)", "cholesky", "generalized_hamiltonian", "sdf", "errvec", "hamiltonian", "dms", "part_energies"]
                    for s, v  in zip(_str, vals):

                        if np.prod(np.shape(v)) > 10:  # don't save numbers 
                            np.savez("numerror/%i_%s.npz"%( n, s), v=v)


            if args.numerror:
                save_plot()
                exit()

        elif args.its == 1:
            vals = jax.jit(dft_iter, backend=args.backend)((args,indxs), 0, vals)
        else:
            vals = jax.lax.fori_loop(0, args.its, lambda i, vals: dft_iter((args,indxs), i, vals), vals)

        eigenvalues   = vals[11]
        eigenvectors  = vals[12]
        energy        = vals[13]
        energies      = vals[ 3]
        dms           = vals[28]
        part_energies = vals[29]

        return energies, energy, eigenvalues, eigenvectors, dms, fixed_hamiltonian, part_energies, L_inv

def density_functional_theory(atom_positions, args, mf_diis_space=9):
    if args.backend == "ipu": mf_diis_space = 9

    if args.generate: global mol

    nuclear_energy    = mol.energy_nuc()
    n_electrons       = mol.nelectron
    n_electrons_half  = n_electrons//2
    hyb               = pyscf.dft.libxc.hybrid_coeff(args.xc, mol.spin)
    N                 = mol.nao_nr()

    mask = np.ones(N)
    mask[n_electrons_half:] = 0
    if args.float32:
        mask = mask.astype(np.float32)

    # Initialize grid.
    grids            = pyscf.dft.gen_grid.Grids(mol)
    grids.level      = args.level
    grids.build()
    coords           = grids.coords
    weights          = grids.weights
    weights          = weights
    ao              = mol.eval_gto('GTOval_cart_deriv1' if mol.cart else 'GTOval_sph_deriv1', coords, 4)

    # Initialize all (N, N) sized matrices (using PySCF or Jax depending on args).
    density_matrix  = np.array(minao(mol)).reshape(N, N)

    kinetic         = mol.intor_symmetric('int1e_kin'). reshape(N, N)
    nuclear         = mol.intor_symmetric('int1e_nuc'). reshape(N, N)
    overlap         = mol.intor_symmetric('int1e_ovlp').reshape(N, N)

    if (args.backend == "cpu" or args.ipumult) and not args.seperate: electron_repulsion = mol.intor("int2e_sph")
    else: electron_repulsion = 0.

    # Turns generalized eigenproblem into eigenproblem.
    fixed_hamiltonian   = kinetic + nuclear
    L_inv               = np.linalg.inv(np.linalg.cholesky(overlap.astype(np.float64)))

    if args.generate:
        __config__.dft_rks_RKS_grids_level = args.plevel

        rng = range(len(args.smiles))
        if args.gdb > 0:
            id, num = args.split
            num_mols = len(args.smiles) // num
            print(len(args.smiles))
            rng  = list(range(id*num_mols, (id+1)*num_mols))
            print(">>> ", min(rng), max(rng))

        if args.save:
            suffix = "_%i_%i"%(min(rng), max(rng))
            if args.fname != "":
                os.makedirs("data/generated/%s/"%(args.fname), exist_ok=True)

                folders = os.listdir("data/generated/%s/"%args.fname)
                print("_%i_%i"%(min(rng), max(rng)) )

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

                  rng = range(len(args.smiles))


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
        do_compute_jit = jax.jit(_do_compute, device=device_1, static_argnums=_do_compute_static_argnums)
        if args.seperate:
            device_2 = jax.devices("ipu")[1]
            print(device_2)
            compute_integrals = jax.jit(compute_integrals_2, static_argnums=(2,3,4,5,6,7), device=device_2)

        pbar = tqdm(rng)
        init_int = True

        # figure out largest grid size by looking at last molecule.
        if True:
            for j in range(1, len(args.smiles)):
                i = rng[-j]

                smile = args.smiles[i]
                atoms = [a for a in list(smile.upper()) if a == "C" or a == "N" or a == "O" or a == "F"]

                b = Chem.MolFromSmiles(smile)
                if not args.nohs: b = Chem.AddHs(b, explicitOnly=False)

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
                grids            = pyscf.dft.gen_grid.Grids(mol)
                grids.level      = args.level
                grids            = build_grid(grids)
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

        ptable = Chem.GetPeriodicTable()
        for count, i in enumerate(pbar):

            # Having the following code in start of for loop allows code to start preparing next molecule asynchronously due to jax asynch dispatch.
            if True:

                if args.gdb == 1337:
                    conformers = [args.qm9["pos"].values[i][0]]*3
                    atoms = [ptable.GetElementSymbol( n) for n in args.qm9["z"].values[i][0].tolist() ]
                    num_hs = len([a for a in atoms if a == "H"])

                else:
                    times.append(time.perf_counter())
                    smile = args.smiles[i]
                    print("[%s]"%smile)
                    atoms = [a for a in list(smile.upper()) if a == "C" or a == "N" or a == "O" or a == "F"]

                    b = Chem.MolFromSmiles(smile)

                    if not args.nohs: b = Chem.AddHs(b, explicitOnly=False)

                    embed_result = AllChem.EmbedMultipleConfs(b, numConfs=args.num_conformers, randomSeed=args.randomSeed)
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
            vals = []

            for conformer_num, conformer in enumerate(conformers):


                if vals != []:
                    prev_mol = mol
                    old_nuclear_energy = nuclear_energy
                    prev_smile = smile
                    prev_locs = locs
                    prev_atoms = atoms

                times.append(time.perf_counter())
                locs = conformer * angstrom_to_bohr
                if args.rattled_std != 0:
                    print(locs)
                    locs += np.random.normal(0, args.rattled_std, locs.shape)
                    print(locs)

                times.append(time.perf_counter())
                atom_string, string = get_atom_string(" ".join(atoms), locs)
                times.append(time.perf_counter())

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

                times.append(time.perf_counter())

                if (args.backend == "cpu" or args.ipumult) and not args.seperate: electron_repulsion = mol.intor('int2e_sph').reshape(N, N, N, N)
                else:                     electron_repulsion = 0

                # Initialize grid.
                if conformer_num == 0 or True:
                    times.append(time.perf_counter())
                    grids            = pyscf.dft.gen_grid.Grids(mol)
                    grids.level      = args.level
                    times.append(time.perf_counter())
                    grids            = build_grid(grids)
                times.append(time.perf_counter())
                coords          = grids.coords
                weights         = grids.weights
                ao              = mol.eval_gto('GTOval_cart_deriv1' if mol.cart else 'GTOval_sph_deriv1', coords, 4)
                times.append(time.perf_counter())
                kinetic         = mol.intor_symmetric('int1e_kin'). reshape(N, N)
                times.append(time.perf_counter())
                nuclear         = mol.intor_symmetric('int1e_nuc'). reshape(N, N)
                times.append(time.perf_counter())
                overlap         = mol.intor_symmetric('int1e_ovlp').reshape(N, N)
                times.append(time.perf_counter())
                fixed_hamiltonian   = kinetic + nuclear
                if vals != []:
                    L_inv_prev = L_inv
                    homo_prev = homo
                    lumo_prev = lumo

                lumo = n_electrons_half
                homo = lumo - 1
                try:
                    if args.choleskycpu:
                        L_inv = np.linalg.inv(np.linalg.cholesky(overlap.astype(np.float64)))
                    else:
                        L_inv = None
                except:
                    print("MATRIX NOT POSITIVE DEFINITE, SKIPPING MOLECULE: ", smile)
                    continue
                times.append(time.perf_counter())

                # Reuse minao across conformers.
                if conformer_num == 0: init_density_matrix = hf.init_guess_by_minao(mol)
                if np.sum(np.isnan(density_matrix)) > 0 or density_matrix.shape != kinetic.shape: density_matrix = hf.init_guess_by_minao(mol)
                density_matrix = init_density_matrix
                times.append(time.perf_counter())

                weights = np.pad(weights, (0, pad-weights.shape[0]))
                coords  = np.pad(coords, ((0, pad-weights.shape[0]), (0, 0)))
                ao      = np.pad(ao, ((0, 0), (0, pad-ao.shape[1]), (0, 0)))

                times.append(time.perf_counter())
                _input_floats, _input_ints = prepare_int_floats(mol)
                times.append(time.perf_counter())

                if args.backend != "cpu":
                    if _input_floats.shape[0] != 400:
                        _input_floats = np.concatenate((_input_floats, np.zeros((1, 400-_input_floats.shape[1]))), axis=1)

                if args.seperate:
                    if init_int: _input_floats, _input_ints, _tuple_ijkl, _shapes, _sizes, _counts, indxs, indxs_inv, num_calls = prepare_integrals_2_inputs(mol)
                    init_int = False
                    if _input_floats.shape[0] != 400:
                        _input_floats = np.concatenate((_input_floats, np.zeros((1, 400-_input_floats.shape[1]))), axis=1)
                    times.append(time.perf_counter())
                    electron_repulsion, cycles_start, cycles_stop = compute_integrals( _input_floats, _input_ints, _tuple_ijkl, _shapes, _sizes, _counts, tuple(indxs_inv), args.threads_int)
                    times.append(time.perf_counter())

                times.append(time.perf_counter())

                if args.save and vals != []:
                    times.append(time.perf_counter())
                    energies_, energy, eigenvalues, eigenvectors, dms, _fixed_hamiltonian, part_energies, _L_inv_prev  = [np.asarray(a).astype(np.float64) for a in vals[-1]]
                    times.append(time.perf_counter())

                    if not args.choleskycpu: L_inv_prev = _L_inv_prev #

                    e = np.zeros(energies_.shape)
                    for i in range(energies_.shape[0]):
                        density_matrix = dms[i,0].reshape(-1)
                        vj             = dms[i,1].reshape(-1)
                        E_coulomb      = np.sum( (density_matrix) * vj) * .5
                        e[i]           = part_energies[i] + np.dot(_fixed_hamiltonian.reshape(-1) , dms[i, 0].reshape(-1)) + E_coulomb + old_nuclear_energy

                    _N = int(np.sqrt(density_matrix.shape[0]))
                    density_matrix = density_matrix.reshape(_N, _N)
                    energy = np.mean(e[-5:])

                    times.append(time.perf_counter())

                    try:
                        mo_energy_us = np.linalg.eigvalsh(L_inv_prev @ dms[-1, 2].reshape(_N,_N) @ L_inv_prev.T)
                    except:
                        try:
                            mo_energy_us = np.linalg.eigh(L_inv_prev @ dms[-1, 2].reshape(_N,_N) @ L_inv_prev.T)[0]
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

                        pyscf_homo = mf.mo_energy[homo]*hartree_to_eV
                        pyscf_lumo = mf.mo_energy[lumo]*hartree_to_eV
                        pyscf_hlgap = np.abs(pyscf_homo - pyscf_lumo )

                        pyscf_energies = np.array(pyscf_energies)

                        if args.pyscf:
                            np.set_printoptions(suppress=True)
                            print(pyscf_energy*hartree_to_eV)
                            print(0.034)

                    times.append(time.perf_counter())  #21
                    us_homo = mo_energy_us[homo_prev]*hartree_to_eV
                    us_lumo = mo_energy_us[lumo_prev]*hartree_to_eV
                    us_hlgap = np.abs(us_homo - us_lumo)
                    if args.verbose:
                        print("jaxdft_hlgap\t", us_hlgap)
                        print("pyscf_hlgap\t", pyscf_hlgap )
                        print("error\t\t", np.abs(us_hlgap-pyscf_hlgap))

                        print(energy*hartree_to_eV)
                        print(pyscf_energy*hartree_to_eV)
                        print(e[-5:]*hartree_to_eV)
                        print(np.array(pyscf_energies[-5:])*hartree_to_eV)

                    if args.checkc and conformer_num > 0:

                        hlgaps_us = []
                        hlgaps_pyscf = []
                        dm_diff = []

                        fig, ax = plt.subplots(1,3, figsize=(10,4))
                        energies = e.reshape(-1)*hartree_to_eV
                        ax[0].plot(energies, label="energy us")
                        ax[0].plot(pyscf_energies*hartree_to_eV, label="energy pyscf")
                        ax[0].legend()
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
                        dct = {
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

                        times.append(time.perf_counter())
                        if not os.path.isfile(filename):  pd.DataFrame(dct).to_csv(filename, mode='a', header=True, compression="gzip")
                        else:  pd.DataFrame(dct).to_csv(filename, mode='a', header=False, compression="gzip")

                if vals != []:
                    times.append(time.perf_counter())
                    times = np.array(times)
                    times = np.around((times[1:] - times[:-1])*1000, 1)
                    pbar.set_description("[%i / %i] Hs=%5i "%(conformer_num, len(conformers), num_hs) +  "%10f "%energy + " ".join([str(a) for a in times.tolist() + [np.around(np.sum(times), 1)]])  + " [%i ; %i]"%(embedded, not_embedded))

                vals  = []
                times = [time.perf_counter()]
                if np.sum(np.isnan(density_matrix)) > 0 or density_matrix.shape != kinetic.shape:
                    density_matrix  = np.array(minao(mol))
                dcargs_names = (
                  'its', 'backend', 'ipumult', 'seperate','threads_int',
                  'intv', 'skiperi', 'randeri', 'float16', 'float32', 'nan', 'forloop',
                  'sk', 'xc', 'threads', 'multv', 'debug',
                  'skipdiis', 'skipeigh', 'geneigh', 'density_mixing'
                )

                dcargs = namedtuple('dcargs', dcargs_names)(*(args.__dict__[a] for a in dcargs_names))
                if not args.forloop:
                    if not args.skip_minao: density_matrix  = np.array(minao(mol))
                    vals.append( do_compute_jit( density_matrix, kinetic, nuclear, overlap, ao, electron_repulsion, weights, coords, nuclear_energy, 0, mf_diis_space, N, hyb , mask, _input_floats, _input_ints, dcargs, L_inv)  )
                else:
                    # make this into a flag.
                    if not args.skip_minao: density_matrix  = np.array(minao(mol))
                    vals.append(_do_compute( density_matrix, kinetic, nuclear, overlap, ao, electron_repulsion, weights, coords, nuclear_energy, 0, mf_diis_space, N, hyb , mask, _input_floats, _input_ints, dcargs, L_inv) )

                times.append(time.perf_counter())


                if args.profile:
                    exit()


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


        vals = jax.jit(_do_compute, static_argnums=_do_compute_static_argnums, device=device_1) ( density_matrix, kinetic, nuclear, overlap,
                                                                                       ao, electron_repulsion, weights, coords, nuclear_energy,
                                                                                       0,
                                                                                       mf_diis_space, N, hyb ,
                                                                                       mask, input_floats, input_ints, args, L_inv)

        energies_, energy, eigenvalues, eigenvectors, dms, fixed_hamiltonian, part_energies, _  = [np.asarray(a).astype(np.float64) for a in vals]
        print(density_matrix.dtype)
        e = np.zeros(energies_.shape)
        for i in range(energies_.shape[0]):
            density_matrix = dms[i,0]
            vj = dms[i,1]
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
                                                                                       mask, input_floats, input_ints, args, L_inv)

        print([a.dtype for a in vals])
        energies_, energy, eigenvalues, eigenvectors, dms, fixed_hamiltonian, part_energies  = [np.asarray(a).astype(np.float64) for a in vals]
        e = np.zeros(energies_.shape)
        for i in range(energies_.shape[0]):
            density_matrix = dms[i,0]
            vj = dms[i,1]
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

def kahan_dot(x, y, sort=False): # more stable dot product;
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

def dft_iter(args_indxs, cycle, val ):
    args, indxs = args_indxs

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
        eigvects = _eigh(generalized_hamiltonian )[1]

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
        density_matrix = jax.lax.select(cycle<=_V.shape[0], density_matrix, (old_dm+density_matrix)/2)

    d = args.float32
    density_matrix, eigvects = f(density_matrix, d), f(eigvects, d)


    if type(electron_repulsion) == type([]):
        if args.float32: E_xc, V_xc, E_coulomb, vj, vk = xc((args, indxs), density_matrix.astype(np.float32), dms.astype(np.float32), cycle, ao.astype(np.float32), electron_repulsion, weights.astype(np.float32), vj.astype(np.float32), vk.astype(np.float32), hyb, num_calls)
    else:
        if args.float32: E_xc, V_xc, E_coulomb, vj, vk = xc((args, indxs), density_matrix.astype(np.float32), dms.astype(np.float32), cycle, ao.astype(np.float32), electron_repulsion.astype(np.float32), weights.astype(np.float32), vj.astype(np.float32), vk.astype(np.float32), hyb, num_calls)
        else: E_xc, V_xc, E_coulomb, vj, vk = xc((args, indxs), density_matrix.astype(np.float64), dms.astype(np.float64), cycle, ao.astype(np.float64), electron_repulsion.astype(np.float64), weights.astype(np.float64), vj.astype(np.float64), vk.astype(np.float64), hyb, num_calls)

    if type(part_energies) == type(jnp.array(1)): part_energies = part_energies.at[cycle].set(  E_xc )
    else: part_energies[cycle] =E_xc


    # Added dynamic_update_slice to optimize compiler layout.
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


def xc(args_indxs, density_matrix, dms, cycle, ao, electron_repulsion, weights, vj, vk, hyb, num_calls):
    # the f notation below allows testing the error when running entire dft in f64 except a single operation in f32.
    args, indxs = args_indxs

    n = density_matrix.shape[0]

    switch = args.float32

    d = 0 in args.sk
    density_matrix, ao = f(density_matrix, d), f(ao, d)
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

    d = switch
    rho, ao0dm, ao = f(rho, d), f(ao0dm, d), f(ao, d)
    d = 2 in args.sk
    rho = f(rho, d)
    #rho                = jnp.concatenate([jnp.clip(rho[:1], CLIP_RHO_MIN, CLIP_RHO_MAX), rho[1:4]*2]) # moved inside b3lyp
    d = switch
    rho = f(rho, d)

    d = 3 in args.sk
    rho = f(rho, d)
    if args.xc == "b3lyp": E_xc, vrho, vgamma = b3lyp(rho, EPSILON_B3LYP)
    elif args.xc == "lda":
        E_xc, vrho, vgamma = lda(rho,   EPSILON_B3LYP)
        assert False, "-xc lda is not not supported. "
    else: E_xc, vrho, vgamma = b3lyp(rho, EPSILON_B3LYP)
    d = switch
    E_xc, vrho, vgamma, rho = f(E_xc, d), f(vrho, d), f(vgamma, d), f(rho,d)

    d = 5 in args.sk
    rho, weights, E_xc = f(rho, d), f(weights, d), f(E_xc, d)
    E_xc =  jnp.sum( rho[0] * weights *  E_xc )
    d = switch
    rho, weights, E_xc = f(rho, d), f(weights, d), f(E_xc, d)

    d = 6 in args.sk
    vrho, vgamma, rho, weights = f(vrho, d), f(vgamma, d), f(rho, d), f(weights, d)
    weird_rho = (jnp.concatenate([vrho.reshape(1, -1)*.5, 4*vgamma*rho[1:4]], axis=0) * weights )
    d = switch
    vrho, vgamma, rho, weights = f(vrho, d), f(vgamma, d), f(rho, d), f(weights, d)

    d = 7 in args.sk
    ao, weird_rho = f(ao, d), f(weird_rho, d)
    n, p = weird_rho.shape
    V_xc = jnp.sum( (ao * weird_rho.reshape(n, p, 1)), axis=0)
    d = switch
    V_xc, ao, weird_rho = f(V_xc, d), f(ao, d), f(weird_rho, d)

    d = 8 in args.sk
    ao, V_xc = f(ao, d), f(V_xc, d)
    V_xc      = ao[0].T @ V_xc
    d = switch
    ao, V_xc = f(ao, d), f(V_xc, d)

    d = 9 in args.sk
    V_xc = f(V_xc, d)
    V_xc      = V_xc + V_xc.T
    d = switch
    V_xc = f(V_xc, d)


    if not args.skiperi:
        d = num_calls.size
        c = 1

        if args.backend == "ipu" and not args.ipumult or args.seperate:
            _tuple_indices, _tuple_do_lists, _N = prepare_ipu_direct_mult_inputs(num_calls.size , mol)

            ipu_vj, ipu_vk = jax.jit(ipu_direct_mult, backend="ipu", static_argnums=(2,3,4,5,6,7,8,9))(
                                                electron_repulsion,
                                                density_matrix,
                                                _tuple_indices,
                                                _tuple_do_lists, _N, num_calls.size,
                                                tuple(indxs.tolist()),
                                                tuple(indxs.tolist()),
                                                int(args.threads),
                                                v=int(args.multv)
                                                )

            vj         = ipu_vj
            vk         = ipu_vk*hyb

        else:
            d = density_matrix.shape[0]
            c = 1

            d = 11 in args.sk
            density_matrix, E = f(density_matrix, d), f(electron_repulsion, d),

            d = density_matrix.shape[0]
            vj = jnp.sum(E.reshape(d**2, d**2) * density_matrix.reshape(1, -1), axis=1).reshape(d,d)

            d = switch
            vj, density_matrix = f(vj, d), f(density_matrix, d)

            d = 12 in args.sk
            density_matrix, E = f(density_matrix, d), f(electron_repulsion, d),
            d = density_matrix.shape[0]
            vk = jnp.sum(E.transpose(1,2,0,3).reshape(d**2, d**2) * density_matrix.reshape(1, -1), axis=1).reshape(d,d)*jnp.asarray(hyb, dtype=E.dtype)
            d = switch
            vk, density_matrix = f(vk, d), f(density_matrix, d)


    else:
        Z = jnp.empty(density_matrix.shape)
        vj, vk = Z, Z

    d = 13 in args.sk
    V_xc, vj, vk = f(V_xc, d), f(vj, d), f(vk, d)
    vj_m_vk = vj/c - vk/2
    V_xc      = (V_xc+ vj_m_vk )
    d = switch
    V_xc, vj, vk = f(V_xc, d), f(vj, d), f(vk, d)


    d = 14 in args.sk
    density_matrix, E_xc, vk = f(density_matrix, d), f(E_xc, d), f(vk, d)
    if args.float32 and not args.backend == "ipu": E_xc      = E_xc - kahan_dot(density_matrix.reshape(-1) , vk.T.reshape(-1)) *( .5 * .5)
    else: E_xc      -= jnp.sum(density_matrix * vk.T) * .5 * .5
    d = switch
    density_matrix, E_xc, vk = f(density_matrix, d), f(E_xc, d), f(vk, d)

    E_coulomb  = 0 # gets computed elsewhere.

    return E_xc, V_xc, E_coulomb, vj, vk


def _eigh(x):
    if args.backend == "ipu" and x.shape[0] >= 6:
        t0 = time.time()
        print("tracing ipu eigh (%s): "%str(x.shape))
        from tessellate_ipu.linalg import ipu_eigh
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
        eigvals, eigvects = jnp.linalg.eigh(f(x, args.float32))

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
    mask = jnp.where(np.arange(_V.shape[0]) < jnp.minimum(cycle+1, _V.shape[0]),
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

    if args.backend == "ipu":
        #c               = pinv( masked_mf_diis_H )[0, :]
        c               = pinv0( masked_mf_diis_H )
        #c               = jnp.linalg.pinv( masked_mf_diis_H )[0, :]
    else:
        c = jnp.linalg.pinv(f(masked_mf_diis_H, args.float32))[0, :]


    scaled_H         = _H[:nd] * c[1:].reshape(nd, 1)
    hamiltonian      = jnp.sum( scaled_H, axis=0 ).reshape(hamiltonian.shape)

    return hamiltonian, _V, _H, mf_diis_H, errvec


def pinv(a):  # take out first row
    #cond = 10. * 9 * 1.1920929e-07
    cond =  9*1.1920929e-07
    #cond =  1.1920929e-07
    vals, vect = _eigh ( a )
    return (vect @ jnp.diag(jnp.where( jnp.abs(vals) > cond, 1/vals, 0)) @ vect.T)

def pinv0(a):  # take out first row
    #cond = 10. * 9 * 1.1920929e-07
    cond =  9*1.1920929e-07
    #cond =  1.1920929e-07
    vals, vect = _eigh ( a )
    c = vect @ ( jnp.where( jnp.abs(vals) > cond, 1/vals, 0) * vect[0, :])
    return c

def recompute(args, molecules, id, num, our_fun, str="", atom_string=""):
  g_float32 = args.float32
  g_ipu = args.backend == 'ipu'

  t0 = time.time()

  if str == "":
    atoms = molecules["atom_string"][id]
    locs  = molecules["atom_locations"][id]*angstrom_to_bohr

    atom_string, str = get_atom_string(atoms, locs)

  mol = Mole()

  mol.build(atom=str, unit="Bohr", basis=args.basis, spin=args.spin, verbose=0)

  print("\t", atom_string, end="")

  if not args.skipus:
    energies, our_energy, our_hlgap, t_us, t_main_loop, us_hlgap = our_fun(str, args)

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

    mf.diis_space = 9

    repeats = 1
    if args.benchmark: repeats = 3


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
    pyscf_energy, pyscf_hlgap, t_pyscf = -1, -1, -1

  if molecules is not None: pcq_hlgap = molecules["hlgap"][id]
  else: pcq_hlgap = -1

  print("UNITS IS [eV]!")
  print("pyscf:\t\t%15f"%pyscf_energy)
  print("us:\t\t%15f"%our_energy)
  print("mus:\t\t%15f"%np.mean(energies[-10:]))
  print("diff:\t\t%15f"%np.abs(pyscf_energy-our_energy))
  print("mdiff:\t\t%15f"%np.abs(pyscf_energy-np.mean(energies[-10:])), np.std(energies[-10:]))
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

_plot_title = f"Created with:  python {' '.join(sys.argv)}"
def save_plot():
    import numpy as np 
    import matplotlib.pyplot as plt 
    import matplotlib
    import os 
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

    import seaborn as sns
    sns.set_theme()
    sns.set_style("white")

    vals = []

    def prepare(val): 
        val = np.abs(val[val == val])
        val[np.logical_and(val<1e-15, val!=0)] = 2e-15 # show the ones that go out of plot 
        val[val==0] = 1e-17 # remove zeros. 
        return val 

    xticks = []
    xticklabels = []

    iterations = 20
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    os.makedirs("images/num_error/", exist_ok=True)

    for outer_num, i in enumerate(range(iterations)):
        skip = 0 
        print("figure [%i / %i]"%(i+1, iterations))
        plt.cla()
        plt.title("[Iterations %i] \n"%(i+1) + _plot_title)
        files = sorted([a for a in os.listdir("numerror/") if "[" not in a and int(a.split("_")[0]) == i and ".jpg" not in a and ".gif" not in a]  )

        for num, file in enumerate(files):
            val= np.load("numerror/"+file)["v"]
            shape = val.shape
            if np.prod(shape) <= 1: 
                skip += 1
                continue 
            
            val = prepare(val)
            val = np.sort(val)
            num_max_dots = 500 

            if val.size > num_max_dots: val= val[::int(val.size)//num_max_dots] 

            ys = -np.ones(val.shape[0])*(num - skip)
            ax.plot([1e-15, 1e18], [ys[0], ys[1]], 'C%i-'%(num%10), lw=10, alpha=0.2)
            ax.plot(val, ys, 'C%io'%(num%10), ms=6, alpha=0.2)

            if i == 0: 
                xticks.append(ys[0])
                xticklabels.append(file.replace(".npz", "").replace("%i_"%i, ""))

        plt.plot( [10**(-10), 10**(-10)], [0, xticks[-1]], 'C7--', alpha=0.6)
        plt.plot( [10**(10), 10**10], [0, xticks[-1]], 'C7--', alpha=0.6)
        plt.plot( [10**(0), 10**0], [0, xticks[-1]], 'C7-', alpha=1)

        for x, label in zip(xticks, xticklabels): 
            ax.text(1e10, x-0.25, label, horizontalalignment='left', size='small', color='black', weight='normal')

        plt.yticks([], [])
        plt.xscale("log")
        plt.xlim([10**(-15), 10**18])
        if i == 0: plt.tight_layout()
        plt.savefig("images/num_error/num_error%i.jpg"%outer_num)

    import imageio 
    writer = imageio.get_writer('images/visualize_DFT_numerics.gif', loop=0, duration=7)
    for i in range(iterations): 
        writer.append_data(imageio.v2.imread("images/num_error/num_error%i.jpg"%i))
    writer.close()
    


mol = None
_str = None
def jax_dft(str, args):
    global mol

    mol = Mole()
    _str = str
    mol.build(atom=str, unit="Bohr", basis=args.basis, spin=args.spin, verbose=0)

    n_electrons       = mol.nelectron
    n_electrons_half  = n_electrons//2
    N                 = mol.nao_nr()

    if args.num == -1 or args.benchmark:
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
        vals = density_functional_theory(atom_positions, args)
        energies, e_tot, mo_energy, mo_coeff, dms, L_inv = [np.asarray(a) for a in vals]
        t = time.time() - t0
        e_tot = energies[-1]*hartree_to_eV

        print(energies*hartree_to_eV)

        if args.benchmark: print("[ it %4f ]"%t)

    mo_occ = jnp.concatenate([jnp.ones(n_electrons_half)*2, jnp.zeros(N-n_electrons_half)])

    d = L_inv.shape[0]
    mo_energy = np.linalg.eigh(L_inv @ np.mean(dms[-1:, 2], axis=0).reshape(d,d) @ L_inv.T)[0]
    lumo            = np.argmin(np.array(mo_occ))
    homo            = lumo - 1
    hl_gap_hartree  = np.abs( mo_energy[homo] - mo_energy[lumo] )
    hlgap           = hartree_to_eV*hl_gap_hartree

    return energies * hartree_to_eV, e_tot, hlgap, t, t, hlgap

def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Arguments for Density Functional Theory. ')
    parser.add_argument('-generate',         action="store_true", help='Enable conformer data generation mode (instead of single point computation). ')
    parser.add_argument('-num_conformers', default=1000, type=int, help='How many rdkit conformers to perfor DFT for. ')
    parser.add_argument('-nohs', action="store_true", help='Whether to not add hydrogens using RDKit (the default adds hydrogens). ')
    parser.add_argument('-verbose', action="store_true")
    parser.add_argument('-choleskycpu',       action="store_true", help='Whether to do np.linalg.inv(np.linalg.cholesky(.)) on cpu/np/f64 as a post-processing step. ')
    parser.add_argument('-resume',       action="store_true", help='In generation mode, dont recompute molecules found in storage folder. ')
    parser.add_argument('-density_mixing',       action="store_true", help='Compute dm=(dm_old+dm)/2')
    parser.add_argument('-skip_minao',       action="store_true", help='In generation mode re-uses minao init across conformers.')
    parser.add_argument('-num',       default=10,          type=int,   help='Run the first "num" test molecules. ')
    parser.add_argument('-id',        default=126,          type=int,   help='Run only test molecule "id". ')
    parser.add_argument('-its',       default=20,          type=int,   help='Number of Kohn-Sham iterations. ')
    parser.add_argument('-step',      default=1,           type=int,   help='If running 1000s of test cases, do molecules[args.skip::args.step]]')
    parser.add_argument('-spin',      default=0,           type=int,   help='Even or odd number of electrons? Currently only support spin=0')
    parser.add_argument('-str',       default="",          help='Molecule string, e.g., "H 0 0 0; H 0 0 1; O 1 0 0; "')
    parser.add_argument('-numerror', action="store_true",     help='Save all tensors to debug numerical errors. ')
    parser.add_argument('-ipumult', action="store_true",     help='On IPU do mult using full tensor ERI computed using PySCF (and not our Rys Quadrature implementation). ')
    parser.add_argument('-skippyscf', action="store_true", help='Skip PySCF used for test case by default. ')
    parser.add_argument('-skipus',    action="store_true", help='Skip our code (and only run PySCF). ')
    parser.add_argument('-float32',   action="store_true", help='Whether to use float32 (default is float64). ')
    parser.add_argument('-float16',   action="store_true", help='Whether to use float16 (default is float64). Not supported. ')
    parser.add_argument('-basis',     default="STO-3G",    help='Which basis set to use. ')
    parser.add_argument('-xc',        default="b3lyp",     help='Only support B3LYP. ')
    parser.add_argument('-skip',      default=0,           help='Skip the first "skip" testcases. ', type=int)
    parser.add_argument('-backend',   default="cpu",       help='Which backend to use, e.g., -backend cpu or -backend ipu')

    parser.add_argument('-benchmark', action="store_true", help='Print benchmark info inside our DFT computation. ')
    parser.add_argument('-nan',       action="store_true", help='Whether to throw assertion on operation causing NaN, useful for debugging NaN arrising from jax.grad. ')
    parser.add_argument('-skipdiis',  action="store_true", help='Whether to skip DIIS; useful for benchmarking.')
    parser.add_argument('-skipeigh',  action="store_true", help='Whether to skip eigh; useful for benchmarking.')
    parser.add_argument('-methane',  action="store_true", help='Simulate methane. ')
    parser.add_argument('-H',        action="store_true", help='Simple hydrogen system. ')
    parser.add_argument('-forloop',  action="store_true", help="Runs SCF iterations in python for loop; allows debugging on CPU, don't run this on IPU it will be super slow. ")
    parser.add_argument('-he',       action="store_true", help="Just do a single He atom, fastest possible case. ")
    parser.add_argument('-level',    default=2, help="Level of the grids used by us (default=2). ", type=int)
    parser.add_argument('-plevel',   default=2, help="Level of the grids used by pyscf (default=2). ", type=int)
    parser.add_argument('-C',         default=-1, type=int,  help='Number of carbons from C20 . ')
    parser.add_argument('-gdb',        default=-1, type=int,  help='Which version of GDP to load {10, 11, 13, 17}. ')
    parser.add_argument('-skiperi',         action="store_true", help='Estimate time if eri wasn\'t used in computation by usig (N,N) matmul instead. ')
    parser.add_argument('-randeri',         action="store_true", help='Initialize electron_repulsion=np.random.normal(0,1,(N,N,N,N))')
    parser.add_argument('-save',         action="store_true", help='Save generated data. ')
    parser.add_argument('-fname',    default="", type=str, help='Folder to save generated data in. ')
    parser.add_argument('-multv',    default=2, type=int, help='Which version of our einsum algorithm to use;comptues ERI@flat(v). Different versions trades-off for memory vs sequentiality. ')
    parser.add_argument('-intv',    default=1, type=int, help='Which version to use of our integral algorithm. ')

    parser.add_argument('-randomSeed',       default=43, type=int,  help='Random seed for RDKit conformer generation. ')

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

    parser.add_argument('-sk',  default=[-2], type=int, nargs="+", 
        help='Used to perform a select number of operations in DFT using f32; '+
             'this allows investigating the numerical errors of each individual operation, '+
             'or multiple operations in combination. ')
    parser.add_argument('-debug',  action="store_true", help='Used to turn on/off the f which is used by the above -sk to investigate float{32,64}. ')

    parser.add_argument('-jit',  action="store_true")
    parser.add_argument('-enable64',  action="store_true", help="f64 is enabled by default; this argument may be useful in collaboration with -sk. ")
    parser.add_argument('-rattled_std',  type=float, default=0, help="Add N(0, args.ratled_std) noise to atom positions before computing dft. ")
    parser.add_argument('-profile',  action="store_true", help="Stops script in generation mode after one molecule; useful when using popvision to profile for -backend ipu")
    parser.add_argument('-pyscf',  action="store_true", help="Used to compute with reference implementation. ")
    parser.add_argument('-uniform_pyscf',  default = -1, type=float, help="Use reference implementation PySCF if 'np.random.uniform(0,1,(1))<args.uniform_pyscf'")
    parser.add_argument('-threads',  default=1, type=int, help="Number of threads to use to compute ipu_mult_direct. ")
    parser.add_argument('-threads_int',  default=1, type=int, help="Number of threads to use to do int2e_sph, accepts {1,...,6}. ")
    parser.add_argument('-split',  default=[1, 16], type=int, nargs="+", help='How to split during data generation over multiple POD16s. 7 47 means split dataset into 47 chunks and this IPU handles chunk 7 (zero indexed).')
    parser.add_argument('-limit', default=-1, type=int, help='smiles = args.smiles[:limit]; gdb molecules are sorted by hydrogens, this allows us to take the ones with fewer hydrogens for which DFT is faster. ')
    parser.add_argument('-seperate',  action="store_true", help='Used to seperate electron integral computation from DFT computation over two chips to lower memory consumption. ')
    parser.add_argument('-gname',  default="", type=str, help='Folder name to store generate dataset; useful when using multiple pods to generate. ')
    parser.add_argument('-checkc',  action="store_true" , help='Check convergence; plot energy every iteration to compare against pyscf. ')
    parser.add_argument('-geneigh',  action="store_true" , help='Use generalized eigendecomposition like pyscf; relies on scipy, only works in debug mode with -forloop. ')

    return parser.parse_args(argv)

def process_args(args):

    print("[BASIS]", args.basis)
    args.sk = tuple(args.sk)

    if -1 in args.sk: args.sk = tuple(range(20))

    if args.checkc:
        args.pyscf = True

    if args.pyscf:
        args.verbose = True

    if True:
        # currently tensor scaling is turned off by default.
        args.scale_w = 1
        args.scale_ao = 1
        if args.float32: args.scale_eri = 1
        args.scale_eri = 1
        args.scale_cholesky= 1
        args.scale_ghamil = 1
        args.scale_eigvects = 1

    if args.numerror:
        args.forloop = True
        args.backend = "cpu"
        args.its = 20

    g_ipu = (args.backend == "ipu")

    if not args.backend == "ipu":
        args.seperate = False

    if args.backend == "ipu":  # allows use of cpu float64 in jnp while using float32 on ipu
        args.ipu = True
        args.sk = tuple(range(20))
        args.debug = True

    return args

def main():
    args = get_args()

    print(sys.argv)
    print(natsorted(vars(args).items()) )

    sys.argv = sys.argv[:1]

    print("")

    if args.float32 or args.float16:
        if args.enable64: config.update('jax_enable_x64', True) 

    else:  # float64
        config.update('jax_enable_x64', True)

    if args.nan:
        config.update("jax_debug_nans", True)

    backend = args.backend
    eigh = _eigh

    if args.str != "":
        recompute(args, None, 0, 0, our_fun=jax_dft, str=args.str)

    elif args.gdb > 0:

        if args.gdb != 20 or True:
            t0 = time.time()
            print("loading gdb data")

            if args.gdb == 10: args.smiles = [a for a in open("gdb/gdb11_size10_sorted.csv", "r").read().split("\n")]
            if args.gdb == 9:  args.smiles = [a for a in open("gdb/gdb11_size09_sorted.csv", "r").read().split("\n")]
            if args.gdb == 7:  args.smiles = [a for a in open("gdb/gdb11_size07_sorted.csv", "r").read().split("\n")]
            if args.gdb == 8:  args.smiles = [a for a in open("gdb/gdb11_size08_sorted.csv", "r").read().split("\n")]

            # used as example data for quick testing.
            if args.gdb == 6:  args.smiles = ["c1ccccc1"]*args.num_conformers
            if args.gdb == 5:  args.smiles = ['CCCCC']*args.num_conformers
            if args.gdb == 4:  args.smiles = ['CCCC']*args.num_conformers


            print("DONE!", time.time()-t0)

            print("Length GDB: ", len(args.smiles))

            if args.limit != -1:
                args.smiles = args.smiles[:args.limit]

            for i in range(int(args.id), int(args.id)+1000):
                smile = args.smiles[i]
                smile = smile

                print(smile)

                b = Chem.MolFromSmiles(smile)
                if not args.nohs: b = Chem.AddHs(b, explicitOnly=False)
                atoms = [atom.GetSymbol() for atom in b.GetAtoms()]

                e = AllChem.EmbedMolecule(b)
                if e == -1: continue

                locs = b.GetConformer().GetPositions() * angstrom_to_bohr
                atom_string, string = get_atom_string(" ".join(atoms), locs)

                print(string)
                break

            recompute(args, None, 0, 0, our_fun=jax_dft, str=string)

    elif args.C > 0:
        _str = [
            ["C", ( 1.56910, -0.65660, -0.93640)],
            ["C", ( 1.76690,  0.64310, -0.47200)],
            ["C", ( 0.47050, -0.66520, -1.79270)],
            ["C", ( 0.01160,  0.64780, -1.82550)],
            ["C", ( 0.79300,  1.46730, -1.02840)],
            ["C", (-0.48740, -1.48180, -1.21570)],
            ["C", (-1.56350, -0.65720, -0.89520)],
            ["C", (-1.26940,  0.64900, -1.27670)],
            ["C", (-0.00230, -1.96180, -0.00720)],
            ["C", (-0.76980, -1.45320,  1.03590)],
            ["C", (-1.75760, -0.63800,  0.47420)],
            ["C", ( 1.28780, -1.45030,  0.16290)],
            ["C", ( 1.28960, -0.65950,  1.30470)],
            ["C", ( 0.01150, -0.64600,  1.85330)],
            ["C", ( 1.58300,  0.64540,  0.89840)],
            ["C", ( 0.48480,  1.43830,  1.19370)],
            ["C", (-0.50320,  0.64690,  1.77530)],
            ["C", (-1.60620,  0.67150,  0.92310)],
            ["C", (-1.29590,  1.48910, -0.16550)],
            ["C", (-0.01020,  1.97270, -0.00630)]
        ][:args.C]

        print(_str)
        recompute(args, None, 0, 0, our_fun=jax_dft, str=_str)

    elif args.H: recompute(args, None, 0, 0, our_fun=jax_dft, str=[["H", (0, 0, 0)],
                                                                   ["H", (1, 1, 1)]])
        
    elif args.methane: recompute(args, None, 0, 0, our_fun=jax_dft, str=[["C", (0, 0, 0)],
                                                                         ["H", (0, 0, 1)],
                                                                         ["H", (1, 0, 0)],
                                                                         ["H", (0, 1, 0)],
                                                                         ["H", (1, 1, 0)]])

if __name__ == "__main__":
    main()
