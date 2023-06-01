now=$(date +"%Y-%m-%d %H:%M:%S")
cmd="XLA_IPU_PLATFORM_DEVICE_COUNT=1 taskset -c 0-14 python density_functional_theory.py -gdb 5 -fname testc5631g  -threads 3 -threads_int 6 -generate -save -backend ipu -float32 -level 0 -plevel 0 -its 20 -skip_minao -intv 1  -multv 2  -basis sto3g -choleskycpu -nohs | tee 'experiments/benchmark/$now.sto3g.txt'"
echo $cmd
tmux new-session -d -s 0 "$cmd"

cmd="XLA_IPU_PLATFORM_DEVICE_COUNT=1 taskset -c 30-44 python density_functional_theory.py -gdb 5 -fname testc5631gs  -threads 2 -threads_int 4 -generate -save -backend ipu -float32 -level 0 -plevel 0 -its 20 -skip_minao -intv 0  -multv 2  -basis 631g -choleskycpu -nohs | tee 'experiments/benchmark/$now.631g.txt'"
echo $cmd
tmux new-session -d -s 1 "$cmd"

cmd="XLA_IPU_PLATFORM_DEVICE_COUNT=1 taskset -c 50-74 python density_functional_theory.py -gdb 5 -fname testc5631gs  -threads 2 -threads_int 2 -generate -save -backend ipu -float32 -level 0 -plevel 0 -its 20 -skip_minao -intv 0  -multv 2  -basis 6-31G* -choleskycpu -nohs | tee 'experiments/benchmark/$now.631gs_t1.txt'"
echo $cmd
tmux new-session -d -s 2 "$cmd"


