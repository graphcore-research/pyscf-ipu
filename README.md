WARNING: THIS IS EXPERIMENTAL RESEARCH CODE, NOT A FINALIZED GRAPHCORE PRODUCT!

# PySCF on IPU
Port of PySCF to Graphcore IPU. 

Limitations
- Restricted Kohn Sham DFT ([RKS](https://github.com/pyscf/pyscf/blob/6c815a62bc2e5eae1488a1d0dbe84556dd54b922/pyscf/dft/rks.py#L531), [KohnShamDFT](https://github.com/pyscf/pyscf/blob/6c815a62bc2e5eae1488a1d0dbe84556dd54b922/pyscf/dft/rks.py#L280) and [hf.RHF](https://github.com/pyscf/pyscf/blob/6c815a62bc2e5eae1488a1d0dbe84556dd54b922/pyscf/scf/hf.py#L2044)).
- Number of atomic orbitals less than 70 `mol.nao_nr() <= 70`. 
- Larger numerical errors due to `np.float32` instead of `np.float64`.
- Limited support for `jax.grad(.)`


## Installation

PySCF on IPU requires Python 3.8, and Graphcore SDK 3.2.

Install the base Python requirements:
```bash
pip install -r requirements.txt
```
If running on a CPU machine (laptop or server), just install the standard `jax`:
```bash
pip install jax==0.3.16 jaxlib==0.3.15
```
On IPU machines, please use Graphcore research experimental JAX:
```bash
pip install jax==0.3.16+ipu jaxlib==0.3.15+ipu.sdk320 -f https://graphcore-research.github.io/jax-experimental/wheels.html
pip install git+https://github.com/graphcore-research/jax-ipu-experimental-addons.git@main
```

The following commands may be useful to check the installation.  
```
python density_functional_theory.py -methane -backend cpu # defaults to float64 as used in PySCF
python density_functional_theory.py -methane -backend cpu -float32
python density_functional_theory.py -methane -backend ipu -float32 
```
This will automatically compare our DFT against PySCF for methane `CH4` and report numerical errors. 

## GDB 11 database preparation

The GDB 11 database required for running PySCF on IPU can be found here: https://zenodo.org/record/5172018.
Please download the `gdb11.tgz` file and extract its content in `gdb/` directory:
```bash
wget -p -O ./gdb/gdb11.tgz https://zenodo.org/record/5172018/files/gdb11.tgz\?download\=1
tar -xvf ./gdb/gdb11.tgz --directory ./gdb/
```



## Running

```bash
python density_functional_theory.py -generate -save -fname dataset_name -level 0 -plevel 0 -gdb 9 -backend cpu -float32
```
