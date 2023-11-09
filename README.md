:red_circle: :warning: **Experimental and non-official Graphcore product** :warning: :red_circle:

[![arXiv](https://img.shields.io/badge/arXiv-2311.01135-b31b1b.svg)](https://arxiv.org/abs/2311.01135)
[![QM1B figshare+](https://img.shields.io/badge/figshare%2B-24459376-blue)](https://doi.org/10.25452/figshare.plus.24459376)
[![notebook-tests](https://github.com/graphcore-research/pyscf-ipu/actions/workflows/notebooks.yaml/badge.svg)](https://github.com/graphcore-research/pyscf-ipu/actions/workflows/notebooks.yaml)
[![nanoDFT CLI](https://github.com/graphcore-research/pyscf-ipu/actions/workflows/cli.yaml/badge.svg)](https://github.com/graphcore-research/pyscf-ipu/actions/workflows/cli.yaml)
[![unit tests](https://github.com/graphcore-research/pyscf-ipu/actions/workflows/unittest.yaml/badge.svg)](https://github.com/graphcore-research/pyscf-ipu/actions/workflows/unittest.yaml)
[![pre-commit checks](https://github.com/graphcore-research/pyscf-ipu/actions/workflows/pre-commit.yaml/badge.svg)](https://github.com/graphcore-research/pyscf-ipu/actions/workflows/pre-commit.yaml)

[**Installation guide**](#installation)
| [**Example DFT Computations**](#example-dft-computations)
| [**Generating data**](#generating-new-datasets)
| [**Training SchNet**](#training-schnet-on-qm1b)
| [**QM1B dataset**](qm1b/README.md)


# PySCF on IPU

PySCF-IPU is built on top of the [PySCF](https://github.com/pyscf) package, porting some of the PySCF algorithms to the Graphcore [IPU](https://www.graphcore.ai/products/ipu).


Only a small portion of PySCF is currently ported, specifically Restricted Kohn Sham DFT (based on [RKS](https://github.com/pyscf/pyscf/blob/6c815a62bc2e5eae1488a1d0dbe84556dd54b922/pyscf/dft/rks.py#L531), [KohnShamDFT](https://github.com/pyscf/pyscf/blob/6c815a62bc2e5eae1488a1d0dbe84556dd54b922/pyscf/dft/rks.py#L280) and [hf.RHF](https://github.com/pyscf/pyscf/blob/6c815a62bc2e5eae1488a1d0dbe84556dd54b922/pyscf/scf/hf.py#L2044)).

The package is under active development, to broaden its scope and applicability.  Current limitations are:
- Number of atomic orbitals less than 70 `mol.nao_nr() <= 70`.
- Larger numerical errors due to `np.float32` instead of `np.float64`.
- Limited support for `jax.grad(.)`

## QuickStart

### For ML dataset generation (SynS & ML Workshop 2023)
To generate datasets based on the paper __Repurposing Density Functional Theory to Suit Deep Learning__ [Link](https://icml.cc/virtual/2023/workshop/21476#wse-detail-28485) [PDF](https://syns-ml.github.io/2023/assets/papers/17.pdf) presented at the [Syns & ML Workshop, ICML 2023](https://syns-ml.github.io/2023/), the entry point is the notebook [DFT Dataset Generation](./notebooks/DFT-dataset-generation.ipynb), and the file [density_functional_theory.py](./density_functional_theory.py).


To run the notebook on Graphcore IPU hardware on Paperspace:

[![Run on Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://ipu.dev/YX0jlK)

### For DFT teaching and learning: nanoDFT

We also provide a lightweight implementation of the SCF algorithm, optimized for readability and hackability, in the [nanoDFT demo](notebooks/nanoDFT-demo.ipynb) notebook and in [nanodft](pyscf_ipu/nanoDFT/README.md) folder.


To run the notebook on Graphcore IPU hardware on Paperspace:

[![Run on Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://ipu.dev/ipobmC)



Additional notebooks in [notebooks](notebooks) demonstrate other aspects of the computation.

## Installation

PySCF on IPU requires Python 3.8, [JAX IPU experimental](https://github.com/graphcore-research/jax-experimental), [TessellateIPU library](https://github.com/graphcore-research/tessellate-ipu) and [Graphcore Poplar SDK 3.2](https://www.graphcore.ai/downloads).

We recommend upgrading `pip` to the latest stable release to prepare your enviroment.
```bash
pip install -U pip
```

This project is currently under active development. 
For CPU simulations, we recommend installing `pyscf-ipu` from latest `main` branch as:
```bash
pip install pyscf-ipu@git+https://github.com/graphcore-research/pyscf-ipu
```

and on IPU equipped machines:
```bash
pip install pyscf-ipu[ipu]@git+https://github.com/graphcore-research/pyscf-ipu
```

## Example DFT Computations
The following commands may be useful to check the installation. Each command runs a test-case which compares PySCF against our DFT computation using different options.
```
python density_functional_theory.py -methane -backend cpu # defaults to float64 as used in PySCF
python density_functional_theory.py -methane -backend cpu -float32
python density_functional_theory.py -methane -backend ipu -float32
```
This will automatically compare our DFT against PySCF for methane `CH4` and report numerical errors.


## Generating New Datasets

This section contains an example on how to generate a DFT dataset based on GDB. This is not needed if you just want to train on the QM1B dataset (to be released soon).

Download the `gdb11.tgz` file from https://zenodo.org/record/5172018 and extract its content in `gdb/` directory:
```bash
wget -p -O ./gdb/gdb11.tgz https://zenodo.org/record/5172018/files/gdb11.tgz\?download\=1
tar -xvf ./gdb/gdb11.tgz --directory ./gdb/
```
To utilize caching you need to sort the SMILES strings by the number of hydrogens RDKit adds to them. This means molecule `i` and `i+1` in most cases have the same number of hydrogens which allows our code to reuse/cache the computational graph for DFT. This can be done by running the following Python script:
```
python ./gdb/sortgdb.py ./gdb/gdb11_size09.smi
```
You can then start generating (locally on CPU) a dataset using the following command:
```bash
python density_functional_theory.py -generate -save -fname dataset_name -level 0 -plevel 0 -gdb 9 -backend cpu -float32
```

You can speed up the generation by using IPUs. Please try the [DFT dataset generation notebook](https://ipu.dev/YX0jlK) [![Run on Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://ipu.dev/YX0jlK)


## Training SchNet on [QM1B](qm1b/README.md)

We used PySCF on IPU to generate the [QM1B dataset](qm1b/README.md) with one billion training examples (to be released soon).
See [Training SchNet on QM1B](./schnet_9m/README.md) for an example implementation of a neural network trained on this dataset.

## License

Copyright (c) 2023 Graphcore Ltd. The project is licensed under the [**Apache License 2.0**](LICENSE), with the exception of the folders `electron_repulsion/` and `exchange_correlation/`.

The library is built on top of the following main dependencies:

| Component | Description | License |
| --- | --- | --- |
| [pyscf](https://github.com/pyscf/pyscf) | Python-based Simulations of Chemistry Framework | [Apache License 2.0](https://github.com/pyscf/pyscf/blob/master/LICENSE) |
| [libcint](https://github.com/sunqm/libcint/) | Open source library for analytical Gaussian integrals | [BSD 2-Clause “Simplified” License](https://github.com/sunqm/libcint/blob/master/LICENSE) |
| [xcauto](https://github.com/dftlibs/xcauto) | Arbitrary order exchange-correlation functional derivatives | [MPL-2.0 license](https://github.com/dftlibs/xcauto/blob/master/LICENSE) |


## Cite
Please use the following citation for the pyscf-ipu project:

```
@inproceedings{mathiasen2023qm1b,
  title={Generating QM1B with PySCF $ \_ $\{$$\backslash$text $\{$IPU$\}$$\}$ $},
  author={Mathiasen, Alexander and Helal, Hatem and Klaeser, Kerstin and Balanca, Paul and Dean, Josef and Luschi, Carlo and Beaini, Dominique and Fitzgibbon, Andrew William and Masters, Dominic},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023}
}
```