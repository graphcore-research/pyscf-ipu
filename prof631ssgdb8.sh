XLA_IPU_PLATFORM_DEVICE_COUNT=2 POPLAR_ENGINE_OPTIONS="{\"autoReport.all\": \"true\", \"autoReport.directory\": \"/a/scratch/alexm/research/popvision/poplar/\"}" taskset -c 0-29 python density_functional_theory.py -gdb 8 -fname run16 -split 0 2 -basis 631gs  -threads 1 -generate -save -backend ipu -float32 -level 0 -plevel 0 -its 50 -seperate -pyscf


