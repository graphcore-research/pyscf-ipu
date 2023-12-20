import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pyscf
import optax
from icecream import ic
from exchange_correlation.b3lyp import b3lyp, _b3lyp, vxc_b3lyp
from tqdm import tqdm 
import time 
from transformer import transformer, transformer_init
import pandas as pd 
import math 

cfg, HARTREE_TO_EV, EPSILON_B3LYP, HYB_B3LYP = None, 27.2114079527, 1e-20, 0.2

def T(x): return jnp.transpose(x, (0,2,1))

B, BxNxN, BxNxK = None, None, None

# Only need to recompute: L_inv, grid_AO, grid_weights, H_core, ERI and E_nuc. 
def dm_energy(W: BxNxK, state, normal, nn): 
    if nn: 
        W = jax.vmap(transformer, in_axes=(None, None, 0, 0, 0), out_axes=(0))(cfg, \
            W, state.ao_types, state.pos, state.H_core)
    # using eigh + H_core + L_inv made training more stable. 
    # todo: investigate whether subset is sufficient (e.g. H_core+qr) or we need all. 
    L_inv_Q: BxNxN        = state.L_inv_T @ jnp.linalg.eigh(state.L_inv @ (state.H_core + W) @ state.L_inv_T)[1]      # O(B*N*K^2) FLOP O(B*N*K) FLOP/FLIO
    density_matrix: BxNxN = 2 * (L_inv_Q*state.mask) @ T(L_inv_Q)                            # O(B*N*K^2) FLOP/FLIO 
    E_xc: B               = exchange_correlation(density_matrix, state, normal) # O(B*gsize*N^2) FLOP O(gsize*N^2) FLIO
    diff_JK: BxNxN        = JK(density_matrix, state, normal)                   # O(B*num_ERIs) FLOP O(num_ERIs) FLIO
    energies: B           = E_xc + state.E_nuc + jnp.sum((density_matrix * (state.H_core + diff_JK/2)).reshape(W.shape[0], -1), axis=-1) 
    energy: float         = jnp.sum(energies) 
    return energy, (energies, E_xc, density_matrix, W)

def sparse_mult(values, dm, state, gsize):
    in_ = dm.take(state.cols, axis=0)
    prod = in_*values[:, None]
    return jax.ops.segment_sum(prod, state.rows, gsize)

def exchange_correlation(density_matrix: BxNxN, state, normal):
    _, _, gsize, N = state.grid_AO.shape
    B = density_matrix.shape[0]
    if normal: 
        grid_AO_dm = (state.grid_AO[:, 0] @ density_matrix)         # (B,gsize,N) @ (B, N, N) = O(B gsize N^2)
        rho        = jnp.sum(grid_AO_dm * state.grid_AO, axis=3)    # (B,1,gsize,N) * (B,4,gsize,N) = O(B gsize N)
    else:    
        main: BxGsizexN       = state.main_grid_AO @ density_matrix            # (1, gsize, N) @ (B, N, N) = O(B gsize N^2) FLOPs and O(gsize*N + N^2 +B * gsize * N) FLIOs 
        correction: BxGsizexN = jax.vmap(sparse_mult, in_axes=(0,0,None, None))(state.sparse_diffs_grid_AO, density_matrix, state, gsize)
        rho_a = jnp.einsum("bpij,bqij->bpi", state.grid_AO, main.reshape(B,1,gsize,N)) 
        rho_b = jnp.einsum("bpij,bqij->bpi", state.grid_AO, correction.reshape(B,1,gsize,N))
        rho = rho_a - rho_b

    E_xc       = jax.vmap(_b3lyp, in_axes=(0, None))(rho, EPSILON_B3LYP).reshape(B, gsize)
    E_xc       = jnp.sum(rho[:, 0] * state.grid_weights * E_xc, axis=-1).reshape(B)
    return E_xc 

def JK(density_matrix, state, normal): 
    if normal: 
        J = jnp.einsum('bijkl,bji->bkl', state.ERI, density_matrix) 
        K = jnp.einsum('bijkl,bjk->bil', state.ERI, density_matrix) 
        diff_JK = J - K / 2 * HYB_B3LYP
    else: 
        from sparse_symmetric_ERI import sparse_symmetric_einsum
        diff_JK: BxNxN = jax.vmap(sparse_symmetric_einsum, in_axes=(None, None, 0))(state.nonzero_distinct_ERI[0], state.nonzero_indices[0], density_matrix)
        diff_JK: BxNxN = diff_JK - jax.vmap(sparse_symmetric_einsum, in_axes=(0, None, 0))(state.diffs_ERI, state.indxs, density_matrix)
 
    return diff_JK 

def nao(atom, basis):
    m = pyscf.gto.Mole(atom='%s 0 0 0; %s 0 0 1;'%(atom, atom), basis=basis)
    m.build()
    return m.nao_nr()//2

def batched_state(mol_str, opts, bs, wiggle_num=0, 
                  do_pyscf=True, validation=False,
                  pad_electrons=45, 
                  pad_diff_ERIs=50000,
                  pad_distinct_ERIs=120000,
                  pad_grid_AO=25000,
                  pad_nonzero_distinct_ERI=200000,
                  pad_sparse_diff_grid=200000, 
                  mol_idx=42,
                  ): 
    if opts.wandb: import wandb 

    # pad molecule if using nn. 
    if not opts.nn:  
        pad_electrons, pad_diff_ERIs, pad_distinct_ERIs, pad_grid_AO, pad_nonzero_distinct_ERI, pad_sparse_diff_grid = \
            -1, -1, -1, -1, -1, -1

    if opts.benzene and opts.nn: pad_electrons = 30
    if opts.hydrogens:
        pad_diff_ERIs = 5000
        pad_distinct_ERIs = 20000
        pad_grid_AO = 2200
        pad_nonzero_distinct_ERI = 20000
        pad_sparse_diff_grid = 20000 

    if opts.qm9: 
        pad_electrons=45
        pad_diff_ERIs=120000
        pad_distinct_ERIs=400000
        pad_grid_AO=50000
        pad_nonzero_distinct_ERI=400000
        pad_sparse_diff_grid=400000

    if opts.alanine: 
        multiplier = 1.2 
        pad_electrons=70
        pad_diff_ERIs=int(180000*1.2)
        pad_distinct_ERIs=int(600000*1.2)
        pad_grid_AO=int(70000*1.2)
        pad_nonzero_distinct_ERI=int(600000*1.2)
        pad_sparse_diff_grid=int(1000000*1.2)

    if opts.waters: 
        pad_diff_ERIs, pad_distinct_ERIs, pad_grid_AO, pad_nonzero_distinct_ERI, pad_sparse_diff_grid = [a//3 for a in [ pad_diff_ERIs , pad_distinct_ERIs , pad_grid_AO , pad_nonzero_distinct_ERI , pad_sparse_diff_grid ]]

    mol = build_mol(mol_str, opts.basis)
    pad_electrons = min(pad_electrons, mol.nao_nr())

    t0 = time.time()
    state = init_dft(mol_str, opts, do_pyscf=do_pyscf, pad_electrons=pad_electrons)
    c, w = state.grid_coords, state.grid_weights
    
    # Set seed to ensure different rotation; initially all workers did same rotation! 
    np.random.seed(mol_idx)
    p = np.array(mol_str[0][1])
    states = [state]
    if opts.waters: 
        water1_xyz = np.array([mol_str[i][1] for i in range(0,3)])
        water2_xyz = np.array([mol_str[i][1] for i in range(3,6)])

    if opts.qm9: 
        atoms = np.array([mol_str[i][1] for i in range(0,3)])

    for iteration in range(bs-1):

        if opts.alanine: 
            from rdkit import Chem
            from rdkit.Chem import AllChem
            pdb_file = 'alanine.pdb'
            molecule = Chem.MolFromPDBFile(pdb_file, removeHs=False)
            # tried reading from pdb_block; caused parallel dataloader pickle to break. 
            AllChem.EmbedMolecule(molecule)
            AllChem.UFFOptimizeMolecule(molecule)
            phi_atoms = [4, 6, 8, 14]  # indices for phi dihedral
            psi_atoms = [6, 8, 14, 16]  # indices for psi dihedral'''

            def xyz(atom): return np.array([atom.x, atom.y, atom.z]).reshape(1, 3)
            def get_atom_positions(mol):
                conf = mol.GetConformer()
                return np.concatenate([xyz(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], axis=0)

            str = [mol_str[j][0] for j in range(len(mol_str))]
            pos = np.concatenate([np.array(mol_str[j][1]).reshape(1, 3) for j in range(len(mol_str))])

            # todo: explore larger space than 10 degree on a single bond! 
            angle = float(np.random.uniform(0, 10, 1)) - 5 
            AllChem.SetDihedralDeg(molecule.GetConformer(), *psi_atoms, angle)
            pos = get_atom_positions(molecule)

            for j in range(22): mol_str[j][1] = tuple(pos[j])

            if iteration == 0: 
                from plot import create_rdkit_mol
                import wandb 
                wandb.log({"mol_valid=%s_angle=%f"%(validation, angle): create_rdkit_mol(str, pos[:21]) })

        if opts.waters:  # todo: rotate both water molecules and draw x=phi, y=psi. 
            rotation_matrix = np.linalg.qr(np.random.normal(size=(3,3)))[0]
            center = water2_xyz.mean(axis=0)
            water_xyz = np.dot(water2_xyz - center, rotation_matrix) + center

            mol_str[3][1] = tuple(water_xyz[0])
            mol_str[4][1] = tuple(water_xyz[1])
            mol_str[5][1] = tuple(water_xyz[2])

            if opts.wandb and iteration == 0: 
                from plot import create_rdkit_mol
                import wandb 
                str = [mol_str[j][0] for j in range(len(mol_str))]
                pos = np.concatenate([np.array(mol_str[j][1]).reshape(1, 3) for j in range(len(mol_str))])
                wandb.log({"%s_mol_%i"%({True: "valid", False: "train"}[validation], iteration): create_rdkit_mol(str, pos) })

        elif opts.qm9: 
            # todo: find dihedral to rotate over similar to alanine dipeptide. 

            # rotate first three atoms around their center of mass 
            # this may break molecule; may need to do this around a bond or smth
            #rotation_matrix = np.linalg.qr(np.random.normal(size=(3,3)))[0]
            #center = atoms.mean(axis=0)
            #rotated_atoms = np.dot(atoms - center, rotation_matrix) + center
            mol_str[0][1] = tuple(atoms[0] + np.random.normal(0, opts.wiggle_var, (3)))
            mol_str[1][1] = tuple(atoms[1] + np.random.normal(0, opts.wiggle_var, (3)))
            mol_str[2][1] = tuple(atoms[2] + np.random.normal(0, opts.wiggle_var, (3)))

            if opts.wandb and iteration == 0: 
                from plot import create_rdkit_mol
                import wandb 
                str = [mol_str[j][0] for j in range(len(mol_str))]
                pos = np.concatenate([np.array(mol_str[j][1]).reshape(1, 3) for j in range(len(mol_str))])
                wandb.log({"%s_mol_%i"%({True: "valid", False: "train"}[validation], iteration): create_rdkit_mol(str, pos) })

        # when profiling create fake molecule to skip waiting
        if iteration == 0 or not opts.prof: stateB = init_dft(mol_str, opts, c, w, do_pyscf=do_pyscf and iteration < 3, state=state, pad_electrons=pad_electrons)

        states.append(stateB)

    state = cats(states)
    N = state.N[0]

    # Compute ERI sparsity. 
    nonzero = []
    for e, i in zip(state.nonzero_distinct_ERI, state.nonzero_indices):
        abs = np.abs(e)
        indxs = abs < opts.eri_threshold #1e-10 
        e[indxs] = 0 
        nonzero.append(np.nonzero(e)[0])

    # Merge nonzero indices and prepare (ij, kl).
    # rep is the number of repetitions we include in the sparse representation. 
    #nonzero_indices = np.union1d(nonzero[0], nonzero[1])  
    union = nonzero[0]
    for i in range(1, len(nonzero)):
        union = np.union1d(union, nonzero[i])
    nonzero_indices = union 
    from sparse_symmetric_ERI import get_i_j, num_repetitions_fast
    ij, kl               = get_i_j(nonzero_indices)
    rep                  = num_repetitions_fast(ij, kl)

    batches = 8
    es = []
    for e,i in zip(state.nonzero_distinct_ERI, state.nonzero_indices):
        nonzero_distinct_ERI = e[nonzero_indices] / rep
        remainder            = nonzero_indices.shape[0] % (batches)
        if remainder != 0: nonzero_distinct_ERI = np.pad(nonzero_distinct_ERI, (0,batches-remainder))

        nonzero_distinct_ERI = nonzero_distinct_ERI.reshape(batches, -1)
        es.append(nonzero_distinct_ERI)

    state.nonzero_distinct_ERI = np.concatenate([np.expand_dims(a, axis=0) for a in es])

    i, j = get_i_j(ij.reshape(-1))
    k, l = get_i_j(kl.reshape(-1))

    if remainder != 0:
        i = np.pad(i, ((0,batches-remainder)))
        j = np.pad(j, ((0,batches-remainder)))
        k = np.pad(k, ((0,batches-remainder)))
        l = np.pad(l, ((0,batches-remainder)))
    nonzero_indices = np.vstack([i,j,k,l]).T.reshape(batches, -1, 4).astype(np.int16)

    state.nonzero_indices = nonzero_indices 

    if opts.normal: diff_state = None 
    else: 
        main_grid_AO   = state.grid_AO[:1]
        diffs_grid_AO  = main_grid_AO - state.grid_AO
        rows, cols = np.nonzero(np.max(diffs_grid_AO[:, 0]!=0, axis=0))

        sparse_diffs_grid_AO = diffs_grid_AO[:, 0, rows,cols]

        # use the same sparsity pattern across a batch.
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
        else: 
            # pad ERIs with 0 and indices with -1 so they point to 0. 
            assert diff_indxs.shape[1] == diff_ERIs.shape[2]
            pad = pad_diff_ERIs - diff_indxs.shape[1]
            assert pad > 0, (pad_diff_ERIs, diff_indxs.shape[1])
            state.indxs     = np.pad(diff_indxs, ((0,0), (0, pad), (0, 0)), 'constant', constant_values=(-1))
            state.diffs_ERI = np.pad(diff_ERIs,  ((0,0), (0, 0),   (0, pad))) # pad zeros 

            if opts.wandb: wandb.log({"pad_diff_ERIs": pad/diff_ERIs.shape[2]})

        state.rows=rows
        state.cols=cols

        state.main_grid_AO=main_grid_AO[:1, 0]

        state.sparse_diffs_grid_AO = sparse_diffs_grid_AO
        state.diffs_grid_AO = diffs_grid_AO

        if pad_sparse_diff_grid != -1: 
            assert state.sparse_diffs_grid_AO.shape[1] == state.rows.shape[0]
            assert state.sparse_diffs_grid_AO.shape[1] == state.cols.shape[0]
            pad = pad_sparse_diff_grid - state.rows.shape[0]
            assert pad >= 0, (pad_sparse_diff_grid, state.rows.shape[0])
            state.rows = np.pad(state.rows, (0,pad))
            state.cols = np.pad(state.cols, (0,pad))
            state.sparse_diffs_grid_AO = np.pad(state.sparse_diffs_grid_AO, ((0,0),(0,pad)))

            if opts.wandb: wandb.log({"pad_sparse_diff_grid": pad/state.sparse_diffs_grid_AO.shape[1]})

        #state.grid_AO = state.grid_AO[:1]
        state.nonzero_distinct_ERI = state.nonzero_distinct_ERI[:1]
        state.nonzero_indices = np.expand_dims(state.nonzero_indices, axis=0)
        if pad_distinct_ERIs != -1: 
            assert state.nonzero_distinct_ERI.shape[2] == state.nonzero_indices.shape[2]
            pad = pad_distinct_ERIs - state.nonzero_distinct_ERI.shape[2]
            assert pad > 0, (pad_distinct_ERIs, state.nonzero_distinct_ERI.shape[2])
            state.nonzero_indices      = np.pad(state.nonzero_indices,      ((0,0), (0,0), (0, pad), (0,0)), 'constant', constant_values=(-1))
            state.nonzero_distinct_ERI = np.pad(state.nonzero_distinct_ERI, ((0,0), (0,0),  (0, pad))) # pad zeros 

            if opts.wandb: wandb.log({"pad_distinct_ERIs": pad/state.nonzero_distinct_ERI.shape[2]})

        if pad_grid_AO != -1: 
            assert state.grid_AO.shape[2] == state.grid_weights.shape[1]
            assert state.grid_AO.shape[2] == state.grid_coords.shape[1]
            assert state.grid_AO.shape[2] == state.main_grid_AO.shape[1]
            assert state.grid_AO.shape[2] == state.diffs_grid_AO.shape[2]
            pad = pad_grid_AO - state.grid_AO.shape[2]
            assert pad > 0, (pad_grid_AO, state.grid_AO.shape[2])
            state.grid_AO      = np.pad(state.grid_AO, ((0,0),(0,0), (0,pad), (0,0)))
            state.grid_weights = np.pad(state.grid_weights, ((0,0),(0,pad)))
            state.grid_coords  = np.pad(state.grid_coords, ((0,0),(0,pad),(0,0)))
            state.main_grid_AO = np.pad(state.main_grid_AO, ((0,0),(0,pad),(0,0)))
            state.diffs_grid_AO = np.pad(state.diffs_grid_AO, ((0,0),(0,0),(0,pad),(0,0)))

            if opts.wandb: wandb.log({"pad_grid_AO": pad/state.grid_AO.shape[2]})


        indxs = np.abs(state.nonzero_distinct_ERI ) > 1e-9 
        state.nonzero_distinct_ERI = state.nonzero_distinct_ERI[indxs]
        state.nonzero_indices = state.nonzero_indices[indxs]
        remainder = state.nonzero_indices.shape[0] % batches

        if remainder != 0:
            state.nonzero_distinct_ERI = np.pad(state.nonzero_distinct_ERI, (0,batches-remainder))
            state.nonzero_indices = np.pad(state.nonzero_indices, ((0,batches-remainder), (0,0)))

        state.nonzero_distinct_ERI = state.nonzero_distinct_ERI.reshape(1, batches, -1)
        state.nonzero_indices = state.nonzero_indices.reshape(1, batches, -1, 4)

        if pad_nonzero_distinct_ERI != -1: 
            assert state.nonzero_distinct_ERI.shape[2] == state.nonzero_indices.shape[2]
            pad = pad_nonzero_distinct_ERI - state.nonzero_distinct_ERI.shape[2]
            assert pad >= 0, (pad_nonzero_distinct_ERI, state.nonzero_distinct_ERI.shape[2])
            state.nonzero_distinct_ERI = np.pad(state.nonzero_distinct_ERI, ((0,0),(0,0),(0,pad)))
            state.nonzero_indices = np.pad(state.nonzero_indices, ((0,0),(0,0),(0,pad), (0,0)), 'constant', constant_values=(-1))

            if opts.wandb: wandb.log({"pad_grid_AO": pad/state.grid_AO.shape[2]})


    import copy 
    return copy.deepcopy(state)


def nanoDFT(mol_str, opts):
    print()
    # Initialize validation set. 
    # This consists of DFT tensors initialized with PySCF/CPU.
    np.random.seed(42)

    if opts.wandb: 
        import wandb 
        if opts.alanine:
            wandb.init(project='ndft_alanine')
        else: 
            wandb.init(project='ndft')

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
        n_heads = 12
        n_layers = 24
    if opts.large: 
        d_model= 1280
        n_heads = 12
        n_layers = 36
    if opts.xlarge: 
        d_model= 1600
        n_heads = 12
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

    if opts.nn: 
        #https://arxiv.org/pdf/1706.03762.pdf see 5.3 optimizer 
        
        # try to mimic karpathy as closely as possible ;) 
        # https://github.com/karpathy/nanoGPT/blob/master/train.py 
        # still differs on 
        # [ ] weight initialization 

        def custom_schedule(it, learning_rate=opts.lr, min_lr=opts.lr/10, warmup_iters=2000, lr_decay_iters=600000):
            if it < warmup_iters: return learning_rate * it / warmup_iters
            if it > lr_decay_iters: return min_lr
            decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
            return min_lr + coeff * (learning_rate - min_lr)

        adam = optax.chain(
            optax.clip_by_global_norm(1),
            optax.scale_by_adam(b1=0.9, b2=0.95, eps=1e-12),
            optax.add_decayed_weights(0.1),#, configure_decay_mask(params)),
            optax.scale_by_schedule(custom_schedule),
            optax.scale(-1),
        )

        w = params 

        from torch.utils.data import DataLoader, Dataset
        class OnTheFlyQM9(Dataset):
            # prepares dft tensors with pyscf "on the fly". 
            # dataloader is very keen on throwing segfaults (e.g. using jnp in dataloader throws segfaul). 
            # problem: second epoch always gives segfault. 
            # hacky fix; make __len__ = real_length*num_epochs and __getitem__ do idx%real_num_examples 
            def __init__(self, opts, nao=294, train=True, num_epochs=10**9):
                # only take molecules with use {CNOFH}, nao=nao and spin=0.
                df = pd.read_pickle("alchemy/processed_atom_9.pickle") # spin=0 and only CNOFH molecules 
                if nao != -1: df = df[df["nao"]==nao] 
                df = df.sample(frac=1).reset_index(drop=True)

                if train: self.mol_strs = df["pyscf"].values[:-10]
                else: self.mol_strs = df["pyscf"].values[-10:]
                self.num_epochs = num_epochs
                self.opts = opts 
                self.validation = not train 

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
                if opts.waters: self.mol_strs = [self.waters]
                if opts.alanine: self.mol_strs = mol_str

                if train: self.bs = opts.bs 
                else: self.bs = opts.val_bs

            def __len__(self):
                return len(self.mol_strs)*self.num_epochs

            def __getitem__(self, idx):
                return batched_state(self.mol_strs[idx%len(self.mol_strs)], self.opts, self.bs, wiggle_num=0, do_pyscf=self.validation, validation=False, mol_idx=idx)
        
        qm9 = OnTheFlyQM9(opts, train=True)
        if opts.workers != 0: 
            train_dataloader = DataLoader(qm9, batch_size=1, pin_memory=True, shuffle=False, drop_last=True, num_workers=opts.workers, prefetch_factor=2, collate_fn=lambda x: x[0])
        else: 
            train_dataloader = DataLoader(qm9, batch_size=1, pin_memory=True, shuffle=False, drop_last=True, num_workers=opts.workers,  collate_fn=lambda x: x[0])
        pbar = tqdm(train_dataloader)
        val_qm9 = OnTheFlyQM9(opts, train=False)

    else: 
        states    = [batched_state(mol_str[0], opts, opts.bs, do_pyscf=True)] + [batched_state(mol_str[i], opts, opts.bs, do_pyscf=False)  for i in range(opts.states-1)]
        class DummyIterator:
            def __init__(self, item): self.item = item
            def __iter__(self): return self
            def __next__(self): return self.item
        train_dataloader = DummyIterator(states[0])
        pbar = tqdm(train_dataloader)
        w = states[0].init 
        adam = optax.adabelief(opts.lr)
        summary(states[0])

    vandg = jax.jit(jax.value_and_grad(dm_energy, has_aux=True), backend=opts.backend, static_argnames=("normal", 'nn'))
    valf = jax.jit(dm_energy, backend=opts.backend, static_argnames=("normal", 'nn'))
    adam_state = adam.init(w)

    if opts.wandb: 
        if not opts.nn: total_params = -1 
        wandb.log({'total_params': total_params, 'batch_size': opts.bs, 'lr': opts.lr })


    min_val, min_dm, mins, val_state, valid_str, step = 0, 0, np.ones(opts.bs)*1e6, None, "", 0
    t0, load_time, train_time, val_time, plot_time = time.time(), 0, 0, 0, 0

    states = []
    for iteration, state in enumerate(pbar):
        if iteration == 0: summary(state) 
        dct = {}
        dct["iteraton"] = iteration 
        load_time, t0 = time.time()-t0, time.time()

        states = states[-opts.grad_acc:] + [state] 
        accumulated_grad = None

        if len(states) < 50: print(len(states))
        
        for j, state in enumerate(states):
            if j == 0: _t0 =time.time()
            (val, (vals, E_xc, density_matrix, _W)), grad = vandg(w, state, opts.normal, opts.nn)
            if j == 0: time_step1 = time.time()-_t0
            accumulated_grad = grad if accumulated_grad is None else jax.tree_map(lambda x, y: x + y, accumulated_grad, grad)

        # scale by global batch size. 
        global_batch_size = len(states)*opts.bs
        wandb.log({"global_batch_size": global_batch_size})
        accumulated_grad = jax.tree_map(lambda x: x / global_batch_size, accumulated_grad)

        # plot grad norm 
        if iteration % 10 == 0: 
            for k,v in accumulated_grad.items(): dct[k + "_norm"] = np.linalg.norm(v .reshape(-1) ) 

        updates, adam_state = adam.update(accumulated_grad, adam_state, params)
        w = optax.apply_updates(w, updates)
        train_time, t0 = time.time()-t0, time.time() 

        if not opts.nn: 
            str = "error=" + "".join(["%.7f "%(vals[i]*HARTREE_TO_EV-state.pyscf_E[i]) for i in range(2)]) + " [eV]"
            str += "pyscf=%.7f us=%.7f"%(state.pyscf_E[0]/HARTREE_TO_EV, vals[0])
        else: 
            pbar.set_description("train=".join(["%.5f"%i for i in vals[:2]]) + "[Ha] "+ valid_str + "time=%.1f %.1f %.1f %.1f %.1f"%(load_time, time_step1, train_time, val_time, plot_time))

        if opts.wandb:
            dct["time_load"]  = load_time 
            dct["time_step1"]  = time_step1
            dct["time_train"]  = train_time
            dct["time_val"]  = val_time 
            plot_iteration = iteration % 10 == 0
            for i in range(0, 2):
                if not opts.nn: 
                    dct['train_l%i'%i ] = np.abs(vals[i]*HARTREE_TO_EV-state.pyscf_E[i])
                    dct['train_pyscf%i'%i ] = np.abs(state.pyscf_E[i])
                dct['train_E%i'%i ] = np.abs(vals[i]*HARTREE_TO_EV)
                if plot_iteration: 
                    dct['img/dm%i'%i] = wandb.Image(np.expand_dims(density_matrix[i], axis=-1))
                    dct['img/W%i'%i] = wandb.Image(np.expand_dims(_W[i], axis=-1))

        plot_time, t0 = time.time()-t0, time.time() 

        if opts.nn:# and i % 5 == 0: 
            if val_state is None: val_state = val_qm9[0]
            _, (valid_vals, _, vdensity_matrix, vW) = valf(w, val_state, opts.normal, opts.nn)
            lr = custom_schedule(iteration)
            valid_str = "lr=%.3e"%lr + "val_error=" + "".join(["%.4f "%(valid_vals[i]*HARTREE_TO_EV-val_state.pyscf_E[i]) for i in range(0, 3)]) + " [eV]"
            if opts.wandb: 
                for i in range(0, opts.val_bs):
                    dct['valid_l%i'%i ] = np.abs(valid_vals[i]*HARTREE_TO_EV-val_state.pyscf_E[i])
                    dct['valid_E%i'%i ] = np.abs(valid_vals[i]*HARTREE_TO_EV)
                    dct['valid_pyscf%i'%i ] = np.abs(val_state.pyscf_E[i])
                    dct['img/val_dm%i'%i] = wandb.Image(np.expand_dims(vdensity_matrix[i], axis=-1))
                    dct['img/val_W%i'%i] = wandb.Image(np.expand_dims(vW[i], axis=-1))
                dct["scheduled_lr"] = custom_schedule(iteration)

        wandb.log(dct)
        valid_time, t0 = time.time()-t0, time.time()

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

  # [ ] consider another grid? 
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
      mol, self.atom_grid, self.radi_method, self.level, self.prune
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
    #t0 = time.time()
    mol = build_mol(mol_str, opts.basis)
    if do_pyscf: pyscf_E, pyscf_hlgap, pycsf_forces = reference(mol_str, opts)
    else:        pyscf_E, pyscf_hlgap, pyscf_forces = np.zeros(1), np.zeros(1), np.zeros(1)

    N                = mol.nao_nr()                                 # N=66 for C6H6 (number of atomic **and** molecular orbitals)
    n_electrons_half = mol.nelectron//2                             # 21 for C6H6
    E_nuc            = mol.energy_nuc()                             # float = 202.4065 [Hartree] for C6H6. TODO(): Port to jax.

    from pyscf import dft
    #grids            = pyscf.dft.gen_grid.Grids(mol)
    grids            = DifferentiableGrids(mol)
    grids.level      = opts.level
    #grids.build()
    grids.build(np.concatenate([np.array(a[1]).reshape(1, 3) for a in mol._atom]), state=state)

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
        nonzero_distinct_ERI = mol.intor("int2e_sph", aosym="s8")
        #ERI = [nonzero_distinct_ERI, nonzero_indices]
        #ERI = ERI 
        ERI = np.zeros(1)
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
    )


    return state


def summary(state): 
    if state is None: return 
    print("_"*100)
    total = 0
    for field_name, field_def in state.__dataclass_fields__.items():
        field_value = getattr(state, field_name)
        try: 
            print("%20s %20s %20s"%(field_name,getattr(field_value, 'shape', None), getattr(field_value, "nbytes", None)/10**9))
            total += getattr(field_value, "nbytes", None)/10**9

        except: 
            try: 
                print("%20s %20s %20s"%(field_name,getattr(field_value[0], 'shape', None), getattr(field_value[0], "nbytes", None)/10**9))
                total += getattr(field_value, "nbytes", None)/10**9
            except: 
                print("BROKE FOR ", field_name)

    print("%20s %20s %20s"%("-", "total", total))
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
    # Use dictionary comprehension to iterate over the dataclass fields
    concatenated_fields = {
        field: _cat(getattr(dc1, field), getattr(dc2, field), field)
        for field in dc1.__annotations__
    }
    # Create a new dataclass instance with the concatenated fields
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

def grad(mol, coords, weight, mo_coeff, mo_energy, dm, H):
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
    __config__.dft_rks_RKS_grids_level = opts.level
    mol = build_mol(mol_str, opts.basis)
    mol.max_cycle = 50 
    mf = pyscf.scf.RKS(mol)
    mf.max_cycle = 50 
    mf.xc = "b3lyp5" 
    mf.diis_space = 8
    pyscf_energies = []
    pyscf_hlgaps = []
    lumo         = mol.nelectron//2
    homo         = lumo - 1
    def callback(envs):
        pyscf_energies.append(envs["e_tot"]*HARTREE_TO_EV)
        hl_gap_hartree = np.abs(envs["mo_energy"][homo] - envs["mo_energy"][lumo]) * HARTREE_TO_EV
        pyscf_hlgaps.append(hl_gap_hartree)
        print("PYSCF: ", pyscf_energies[-1] , "[eV]")
    mf.callback = callback
    mf.kernel()
    print("")
    if False: 
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
    mol.build(atom=mol_str, unit="Angstrom", basis=basis_name, spin=0, verbose=0)
    return mol

def reference(mol_str, opts):
    import pickle 
    import hashlib 
    if opts.skip:  return np.zeros(1), np.zeros(1), np.zeros(1) 
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
    import os
    import argparse 

    parser = argparse.ArgumentParser()
    # DFT options 
    parser.add_argument('-basis',   type=str,   default="sto3g")  
    parser.add_argument('-level',   type=int,   default=0)

    # GD options 
    parser.add_argument('-backend', type=str,   default="cpu") 
    parser.add_argument('-lr',      type=float, default=2.5e-4)
    parser.add_argument('-steps',   type=int,   default=100000)
    parser.add_argument('-bs',      type=int,   default=8)
    parser.add_argument('-val_bs',  type=int,   default=8)
    parser.add_argument('-grad_acc',  type=int,   default=16)

    parser.add_argument('-normal',     action="store_true") 
    parser.add_argument('-wandb',      action="store_true") 
    parser.add_argument('-prof',      action="store_true") 
    parser.add_argument('-visualize',  action="store_true") 
    parser.add_argument('-skip',       action="store_true", help="skip pyscf test case") 

    # dataset 
    parser.add_argument('-qm9',        action="store_true") 
    parser.add_argument('-benzene',        action="store_true") 
    parser.add_argument('-hydrogens',        action="store_true") 
    parser.add_argument('-water',        action="store_true") 
    parser.add_argument('-waters',        action="store_true") 
    parser.add_argument('-alanine',        action="store_true") 
    parser.add_argument('-states',         type=int,   default=1)
    parser.add_argument('-workers',        type=int,   default=5) # set to 0 for faster pyscf precompute. 
        # do noise schedule, start small slowly increase 
    parser.add_argument('-wiggle_var',     type=float,   default=0.05, help="wiggle N(0, wiggle_var), bondlength=1.5/30")
    parser.add_argument('-eri_threshold',  type=float,   default=1e-10, help="loss function threshold only")

    # models 
    parser.add_argument('-nn',       action="store_true", help="train nn, defaults to GD") 
    parser.add_argument('-tiny',     action="store_true") 
    parser.add_argument('-small',    action="store_true") 
    parser.add_argument('-base',     action="store_true") 
    parser.add_argument('-medium',     action="store_true") 
    parser.add_argument('-large',     action="store_true") 
    parser.add_argument('-xlarge',     action="store_true") 
    opts = parser.parse_args()
    if opts.tiny or opts.small or opts.base: opts.nn = True 

    args_dict = vars(opts)
    print(args_dict)

    if opts.qm9: 
        df = pd.read_pickle("alchemy/atom_9.pickle")
        df = df[df["spin"] == 0] # only consider spin=0
        mol_strs = df["pyscf"].values

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
    if opts.water: 
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
        mol_strs = [[
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
