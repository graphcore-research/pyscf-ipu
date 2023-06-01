


tmux new-session -d -s 1 "XLA_IPU_PLATFORM_DEVICE_COUNT=2  taskset -c 0-29  python density_functional_theory.py -gdb 9 -fname run16 -split 0 2 -basis sto3g  -threads 1 -generate -save -backend ipu -float32 -level 0 -plevel 0 -its 50 -seperate |& tee tmp/basis1.txt"
tmux new-session -d -s 2 "XLA_IPU_PLATFORM_DEVICE_COUNT=2  taskset -c 30-59  python density_functional_theory.py -gdb 9 -fname run16 -split 0 2 -basis sto-6g  -threads 1 -generate -save -backend ipu -float32 -level 0 -plevel 0 -its 50 -seperate |& tee tmp/basis2.txt "
tmux new-session -d -s 3 "XLA_IPU_PLATFORM_DEVICE_COUNT=2  taskset -c 60-89  python density_functional_theory.py -gdb 9 -fname run16 -split 0 2 -basis 321g  -threads 1 -generate -save -backend ipu -float32 -level 0 -plevel 0 -its 50 -seperate |& tee tmp/basis3.txt "
tmux new-session -d -s 4 "XLA_IPU_PLATFORM_DEVICE_COUNT=2  taskset -c 90-119  python density_functional_theory.py -gdb 9 -fname run16 -split 0 2 -basis 431g  -threads 1 -generate -save -backend ipu -float32 -level 0 -plevel 0 -its 50 -seperate |& tee tmp/basis4.txt "
tmux new-session -d -s 5 "XLA_IPU_PLATFORM_DEVICE_COUNT=2  taskset -c 120-149  python density_functional_theory.py -gdb 9 -fname run16 -split 0 2 -basis 631g  -threads 1 -generate -save -backend ipu -float32 -level 0 -plevel 0 -its 50 -seperate |& tee tmp/basis5.txt "
#tmux new-session -d -s 6 "XLA_IPU_PLATFORM_DEVICE_COUNT=2  taskset -c 120-149  python density_functional_theory.py -gdb 9 -fname test -split 0 256 -basis 631g*  -threads 1 -generate -save -backend ipu -float32 -level 0 -plevel 0 -its 50 |& tee tmp/basis6.txt "
#tmux new-session -d -s 7 "XLA_IPU_PLATFORM_DEVICE_COUNT=2  taskset -c 120-149  python density_functional_theory.py -gdb 9 -fname test -split 0 256 -basis 6-31G**\"  -threads 1 -generate -save -backend ipu -float32 -level 0 -plevel 0 -its 50 |& tee tmp/basis7.txt "
#tmux new-session -d -s 8 "XLA_IPU_PLATFORM_DEVICE_COUNT=2  taskset -c 120-149  python density_functional_theory.py -gdb 9 -fname test -split 0 256 -basis 3-21G*\"  -threads 1 -generate -save -backend ipu -float32 -level 0 -plevel 0 -its 50 |& tee tmp/basis8.txt "
