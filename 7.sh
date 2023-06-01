tmux new-session -d -s 0 "TF_POPLAR_FLAGS=--executable_cache_path=\"_cache/\" taskset -c 0-14  python density_functional_theory.py -gdb 10 -fname $1 -pyscf -split 32 64 -threads 2 -generate -save -backend ipu -float32 -level 0 -plevel 0 -its 50 -ipumult"

for X in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
        v1=$((X*15)) 
        v2=$((X*15 + 14))
        n=$((X + 16+16))

        echo "$v1 $v2"
        tmux new-session -d -s $X "TF_POPLAR_FLAGS=--executable_cache_path=\"_cache/\" taskset -c $v1-$v2 python density_functional_theory.py -gdb 10 -fname $1 -split $n 64 -threads 2 -generate -save -backend ipu -float32 -level 0 -plevel 0 -its 50 -ipumult |& tee tmp/7gentest$X.txt "
done