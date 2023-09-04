import argparse
from natsort import natsorted


def parse_args():

    parser = argparse.ArgumentParser(description="Arguments for Density Functional Theory.")
    parser.add_argument('-generate', action="store_true", help='Enable conformer data generation mode (instead of single point computation). ')
    parser.add_argument('-num_conformers', default=1000, type=int, help='How many rdkit conformers to perfor DFT for. ')
    parser.add_argument('-nohs', action="store_true", help='Whether to not add hydrogens using RDKit (the default adds hydrogens). ')
    parser.add_argument('-verbose', action="store_true")
    parser.add_argument('-choleskycpu', action="store_true", help='Whether to do np.linalg.inv(np.linalg.cholesky(.)) on cpu/np/f64 as a post-processing step. ')
    parser.add_argument('-resume', action="store_true", help='In generation mode, dont recompute molecules found in storage folder. ')
    parser.add_argument('-density_mixing', action="store_true", help='Compute dm=(dm_old+dm)/2')
    parser.add_argument('-skip_minao', action="store_true", help='In generation mode re-uses minao init across conformers.')
    parser.add_argument('-num', default=10,          type=int,   help='Run the first "num" test molecules. ')
    parser.add_argument('-id', default=126,          type=int,   help='Run only test molecule "id". ')
    parser.add_argument('-its', default=20,          type=int,   help='Number of Kohn-Sham iterations. ')
    parser.add_argument('-step', default=1,           type=int,   help='If running 1000s of test cases, do molecules[args.skip::args.step]]')
    parser.add_argument('-spin', default=0,           type=int,   help='Even or odd number of electrons? Currently only support spin=0')
    parser.add_argument('-str', default="",          help='Molecule string, e.g., "H 0 0 0; H 0 0 1; O 1 0 0; "')
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
    parser.add_argument('-skipdiis',  action="store_true", help='Whether to skip DIIS; useful for benchmarking.')
    parser.add_argument('-skipeigh',  action="store_true", help='Whether to skip eigh; useful for benchmarking.')
    parser.add_argument('-methane',  action="store_true", help='Simulate methane. ')
    parser.add_argument('-H',        action="store_true", help='Simple hydrogen system. ')
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

    parser.add_argument('-sk',  default=[-2], type=int, nargs="+", help='Used to perform a select number of operations in DFT using f32; this allows investigating the numerical errors of each individual operation, or multiple operations in combination. ')
    parser.add_argument('-debug',  action="store_true", help='Used to turn on/off the f which is used by the above -sk to investigate float{32,64}. ')
    parser.add_argument('-numerror', action="store_true",     help='Save all tensors to debug numerical errors. ')
    parser.add_argument('-forloop',  action="store_true", help="Runs SCF iterations in python for loop; allows debugging on CPU, don't run this on IPU it will be super slow. ")
    parser.add_argument('-nan',       action="store_true", help='Whether to throw assertion on operation causing NaN, useful for debugging NaN arrising from jax.grad. ')
    parser.add_argument('-geneigh',  action="store_true" , help='Use generalized eigendecomposition like pyscf; relies on scipy, only works in debug mode with -forloop. ')


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

    args = parser.parse_args()

    print("\nArguments passed:")
    max_arg_length = max(len(arg) for arg, value in vars(args).items())
    for arg, value in natsorted(vars(args).items()):
        print(f"{arg:_<{max_arg_length}} {value}")

    return args


# from dataclasses import dataclass

# @dartaclass
# class Config:
