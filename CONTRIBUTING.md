# Contributing to pyscf-ipu

This project is still evolving but at the moment is focused around a high-performance and easily hackable implementation of Gaussian basis set DFT.
We hope this is useful for the generation of large-scale datasets needed
for training machine-learning models. We are interested in hearing any and
all feedback so feel free to raise any questions, bugs encountered, or enhancement requests as [Issues](https://github.com/graphcore-research/pyscf-ipu/issues).

## Setting up a development environment
We recommend using the conda package manager as this can automatically enable
the Graphcore Poplar SDK.  This is particularly useful in VS Code which can automatically
activate the conda environment in a variety of scenarios:
* visual debugging
* running quick experiments in an interactive Jupyter window
* using VS code for Jupyter notebook development.

The following assumes that you have already setup an install of conda and that
the conda command is available on your system path.  Refer to your preferred conda
installer:
* [miniforge installation](https://github.com/conda-forge/miniforge#install)
* [conda installation documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

1. Create a new conda environment with the same python version as you OS.
   For example, on ubuntu 20 use `python=3.8.10`
   ```bash
   conda create -n pyscf-ipu python=3.8.10
   ```

2. Confirm that you have the Poplar SDK installed on your machine and store the location
   in an environment variable.  The following will test that the SDK is found and
   configured correctly:
   ```bash
   export POPLAR_SDK=/path/to/sdk
   source $POPLAR_SDK/enable
   gc-monitor
   ```

3. Activate the environment and store a persistent environment variable for the
   location of the downloaded Poplar SDK. This assumes that
   you have already downloaded the Poplar SDK.  The following example uses an
   environment variable `$POPLAR_SDK` to store the root folder for the SDK.
   ```bash
   conda activate pyscf-ipu
   conda env config vars set POPLAR_SDK=$POPLAR_SDK
   ```

4. You have to reactivate the conda environment to use the `$POPLAR_SDK`
   variable the environment.
   ```bash
   conda deactivate
   conda activate pyscf-ipu
   ```

5. Setup the conda environment to automatically enable the Poplar SDK whenever
   the environment is activated.
   ```bash
   mkdir -p $CONDA_PREFIX/etc/conda/activate.d
   echo "source $POPLAR_SDK/enable" > $CONDA_PREFIX/etc/conda/activate.d/enable.sh
   ```

6. Check that everything is working by once again reactivating the pyscf-ipu
   environment and calling `gc-monitor`:
   ```bash
   conda deactivate
   conda activate pyscf-ipu
   gc-monitor
   ```

7. Install all required packages for developing JAX DFT:
   ```bash
   pip install -e ".[ipu,test]"
   ```

8. Install the pre-commit hooks
   ```bash
   pre-commit install
   ```

9. Create a feature branch, make changes, and when you commit them the
   pre-commit hooks will run.
   ```bash
   git checkout -b feature
   ...
   git push --set-upstream origin feature
   ```
   The last command will prints a link that you can follow to open a PR.


## Testing
Run all the tests using `pytest`
```bash
pytest
```
We also use the nbmake package to check our notebooks work in the `IpuModel` environment.  These checks can also be run on IPU hardware equiped machines e.g.:
```bash
pytest --nbmake --nbmake-timeout=3000 notebooks/nanoDFT-demo.ipynb
```
