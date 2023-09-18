# Contributing to pyscf-ipu


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

2. Activate the environment and store a persistent environment variable for the
   location of the downloaded Poplar SDK. This assumes that
   you have already downloaded the Poplar SDK.  The following example uses an
   environment variable `$POPLAR_SDK` to store the root folder for the SDK.
   ```bash
   conda activate pyscf-ipu
   conda env config vars set POPLAR_SDK=/path/to/poplar/sdk
   ```

3. You have to reactivate the conda environment to use the `$POPLAR_SDK`
   variable the environment.
   ```bash
   conda deactivate
   conda activate pyscf-ipu
   ```

4. Setup the conda environment to automatically enable the Poplar SDK whenever
   the environment is activated.
   ```bash
   mkdir -p $CONDA_PREFIX/etc/conda/activate.d
   echo "source $POPLAR_SDK/enable" > $CONDA_PREFIX/etc/conda/activate.d/enable.sh
   ```

5. Check that everything is working by once again reactivating the pyscf-ipu
   environment and calling `gc-monitor`:
   ```bash
   conda deactivate
   conda activate pyscf-ipu
   gc-monitor
   ```

5. Install all required packages for developing JAX DFT:
   ```bash
   pip install -e ".[ipu,test]"
   ```

6. Install the pre-commit hooks
   ```bash
   pre-commit install
   ```

7. Create a feature branch, make changes, and when you commit them the
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

## Profiling
