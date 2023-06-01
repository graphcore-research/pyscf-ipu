# $1 = name of run
# $2 = basis set
for X in 0 1 2 3 4 5 6 7 
#for X in 7 
do
        v1=$((X*30)) 
        v2=$((X*30 + 29))
        n=$((X + offset))
        echo "data/generated/$1/$2.$X.txt"

        echo "XLA_IPU_PLATFORM_DEVICE_COUNT=2 POPLAR_ENGINE_OPTIONS='{\"autoReport.all\": \"true\", \"autoReport.directory\": \"/a/scratch/alexm/research/popvision/poplar/\"}' taskset -c $v1-$v2 python density_functional_theory.py -gdb 13 -fname $1 -split $n 8 -threads $3 -threads_int $4  -generate -save -backend ipu -float32 -level 0 -plevel 0 -its 20  -multv 2 -seperate -uniform_pyscf 0.001 -basis $2 2>&1 | tee data/generated/$1/$X.$2.txt"

        echo "$v1 $v2"
	mkdir -p "data/generated/$1/"
        tmux new-session -d -s $X "XLA_IPU_PLATFORM_DEVICE_COUNT=2 TF_POPLAR_FLAGS=--executable_cache_path=\"/a/scratch/alexm/research/cache/\" taskset -c $v1-$v2 python density_functional_theory.py -gdb 13 -fname $1 -split $n 8 -threads $3 -threads_int $4  -generate -save -backend ipu -float32 -level 0 -plevel 0 -its 20  -multv 2 -seperate -uniform_pyscf 0.0001 -basis $2 2>&1 | tee data/generated/$1/$X.$2.txt"
        #tmux new-session -d -s $X "XLA_IPU_PLATFORM_DEVICE_COUNT=2 POPLAR_ENGINE_OPTIONS='{\"autoReport.all\": \"true\", \"autoReport.directory\": \"/a/scratch/alexm/research/popvision/poplar/\"}' taskset -c $v1-$v2 python density_functional_theory.py -gdb 9 -fname $1 -split $n 8 -threads $3 -threads_int $4  -generate -save -backend ipu -float32 -level 0 -plevel 0 -its 20  -multv 2 -seperate -uniform_pyscf 0.000 -basis $2 2>&1 | tee data/generated/$1/$X.$2.txt"
done
