# split into 256 => do the easiest first half ~ 10M 
offset=$((8*$2))
echo "$offset"

#tmux new-session -d -s 0 "XLA_IPU_PLATFORM_DEVICE_COUNT=2 TF_POPLAR_FLAGS=--executable_cache_path=\"_cache/\" taskset -c 0-29  python density_functional_theory.py -gdb 11 -fname $1 -pyscf -split $offset 256 -threads 3 -generate -save -backend ipu -float32 -level 0 -plevel 0 -its 50  -seperate"
tmux new-session -d -s 0 "XLA_IPU_PLATFORM_DEVICE_COUNT=2 TF_POPLAR_FLAGS=--executable_cache_path=\"_cache/\" taskset -c 0-29  python density_functional_theory.py -gdb 11 -fname $1 -split $offset 256 -threads 3 -generate -save -backend ipu -float32 -level 0 -plevel 0 -its 50  -seperate"

for X in 1 2 3 4 5 6 7 
do
        v1=$((X*30)) 
        v2=$((X*30 + 29))
        n=$((X + offset))

        echo "$v1 $v2"
        tmux new-session -d -s $X "XLA_IPU_PLATFORM_DEVICE_COUNT=2 TF_POPLAR_FLAGS=--executable_cache_path=\"_cache/\" taskset -c $v1-$v2 python density_functional_theory.py -gdb 11 -fname $1 -split $n 256 -threads 3 -generate -save -backend ipu -float32 -level 0 -plevel 0 -its 50  -seperate |& tee tmp/sep_$offset.gdb118gentest$X.txt "
done
