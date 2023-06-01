# PySCF on IPU

Port of PySCF to Graphcore IPU.

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

## GDB 11 database preparation

The GDB 11 database required for running PySCF on IPU can be downloaded here: https://zenodo.org/record/5172018
Please download the `gdb11.tgz` file and extract its content in `gdb/` directory.

## Running

```bash
python density_functional_theory.py -generate -save -fname dataset_name -level 0 -plevel 0 -gdb 9 -backend cpu -float32
```