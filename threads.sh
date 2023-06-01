#for X in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
for X in 0 1 2 3 4 5 
do
        v1=$((X*10)) # perhaps try to align this with 2 things on each numa node, and hten perhaps numba node 1 is free? 
        v2=$((X*10 + 9))
        t=$((1+X))

        echo "$v1 $v2"
        #tmux new-session -d -s $X "taskset -c $v1-$v2 python density_functional_theory.py -gdb 11 -threads $t -backend ipu -float32 -level 0 -plevel 0 -its 30 |& tee tmp/test$X.txt "
        #tmux new-session -d -s $X "taskset -c $v1-$v2 ./prof.sh density_functional_theory.py -gdb 11 -threads $t -backend ipu -float32 -level 0 -plevel 0 -its 30 |& tee tmp/test$X.txt "
        #tmux new-session -d -s $X "taskset -c $v1-$v2 ./prof.sh density_functional_theory.py -gdb 9 -threads $t -backend ipu -float32 -level 0 -plevel 0 -its 50 |& tee tmp/test$X.txt "
	 #tmux new-session -d -s $X "XLA_IPU_PLATFORM_DEVICE_COUNT=2 TF_POPLAR_FLAGS=--executable_cache_path=\"_cache/\" taskset -c $v1-$v2  python density_functional_theory.py -gdb 11 -fname test  -pyscf -split 0 256  -threads $t  -backend ipu -float32 -level 0 -plevel 0 -its 50  -generate -seperate -save"
	 tmux new-session -d -s $X "XLA_IPU_PLATFORM_DEVICE_COUNT=2 TF_POPLAR_FLAGS=--executable_cache_path=\"_cache/\" taskset -c $v1-$v2  python density_functional_theory.py -gdb 11 -fname test   -split 0 256  -threads $t  -backend ipu -float32 -level 0 -plevel 0 -its 50  -generate -seperate -save"

done

