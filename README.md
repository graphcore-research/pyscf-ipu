WARNING: THIS IS EXPERIMENTAL RESEARCH CODE, NOT A FINALIZED GRAPHCORE PRODUCT!

# PySCF on IPU
Port of PySCF to Graphcore IPU. 

Limitations
- Restricted Kohn Sham DFT (based on [RKS](https://github.com/pyscf/pyscf/blob/6c815a62bc2e5eae1488a1d0dbe84556dd54b922/pyscf/dft/rks.py#L531), [KohnShamDFT](https://github.com/pyscf/pyscf/blob/6c815a62bc2e5eae1488a1d0dbe84556dd54b922/pyscf/dft/rks.py#L280) and [hf.RHF](https://github.com/pyscf/pyscf/blob/6c815a62bc2e5eae1488a1d0dbe84556dd54b922/pyscf/scf/hf.py#L2044)).
- Number of atomic orbitals less than 70 `mol.nao_nr() <= 70`. 
- Larger numerical errors due to `np.float32` instead of `np.float64`.
- Limited support for `jax.grad(.)`

## Installation

PySCF on IPU requires Python 3.8, and Graphcore SDK 3.2.

To run this package on a standard CPU machine (laptop or server ), 
install the base Python requirements:
```bash
pip install -r requirements.txt
```

On IPU machines, please use the IPU requirements file:
```bash
pip install -r requirements_ipu.txt
```
This will configure Graphcore research experimental JAX support in your python environment.

## Example DFT Computations
The following commands may be useful to check the installation. Each command runs a test-case which compares PySCF against our DFT computation using different options. 
```
python density_functional_theory.py -methane -backend cpu # defaults to float64 as used in PySCF
python density_functional_theory.py -methane -backend cpu -float32
python density_functional_theory.py -methane -backend ipu -float32 
```
This will automatically compare our DFT against PySCF for methane `CH4` and report numerical errors. 

## Generating New Datasets
This section contains an example on how to generate a DFT dataset based of GDB. This is not needed if you just want to train on QM1B.  

Download the `gdb11.tgz` file from https://zenodo.org/record/5172018 and extract its content in `gdb/` directory:
```bash
wget -p -O ./gdb/gdb11.tgz https://zenodo.org/record/5172018/files/gdb11.tgz\?download\=1
tar -xvf ./gdb/gdb11.tgz --directory ./gdb/
```
To utilize caching you need to sort the SMILES strings by the number of hydrogens RDKit adds to them. This means molecule `i` and `i+1` in most cases have the same number of hydrogens which allows our code reuse/cache the computational graph for DFT. This can be done by running the cells in 
```
gdb/sortgdb9.ipynb
```
You can then start generating a dataset using the following command:
```bash
python density_functional_theory.py -generate -save -fname dataset_name -level 0 -plevel 0 -gdb 9 -backend cpu -float32
```
You can speed up the generation by using IPUs through https://www.paperspace.com/

## Training SchNet on QM1B

We used PySCF on IPU to generate the QM1B dataset with one billion training examples.
See [Training SchNet on QM1B](./schnet_9m/README.md) for an example implementation of a neural network trained on this dataset. 