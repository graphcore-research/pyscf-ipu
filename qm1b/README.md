# QM1B dataset (to be released soon)

QM1B is a low-resolution DFT dataset generated using [PySCF IPU](https://github.com/graphcore-research/pyscf-ipu). It is composed of one billion training examples containing 9-11 heavy atoms. It was created by taking 1.09M SMILES strings from the [GDB-11 database](https://zenodo.org/record/5172018) and computing molecular properties (e.g. HOMO-LUMO gap) for a set of up to 1000 conformers per molecule.

## Dataset schema

QM1B dataset is stored in the [open-source columnar Apache Parquet format](https://parquet.apache.org/), with the following schema:
* `smile`: The SMILES string taken from GDB11. There are up to 1000 rows (i.e. conformers) with the same SMILES
string.
* `atoms`: String representing the atom symbols of the molecule, e.g. ”COOH”.
* `z`: Integer representation of `atoms` used by SchNet (the atomic numbers).
* `energy`: energy of the molecule computed by PySCF IPU (unit eV).
* `homo`: The energy of the Highest Occupied Molecular Orbital (HOMO) (unit eV).
* `lumo`: The energy of the Lowest occupied Molecular Orbital (LUMO) (unit eV).
* `N`: The number of atomic orbitals for the specific DFT computation (depends on the basis set STO3G).
* `std`: The standard deviation of the energy of the last five iterations of running PySCFIPU, used as
convergence criteria std < 0.01 (unit eV).
* `y`: The HOMO-LUMO Gap (unit eV).
* `pos`: The atom positions (unit Bohr).

## Dataset exploration

Dataset exploration can easily done using Pandas library. For instance, to load the validation set:
```python
import pandas as pd

# 20m entries in the validation set.
print(pd.read_parquet("qm1b_val.parquet").head())
```