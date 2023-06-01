for X in 1 2 3 4 5 6 7 8 9 10 11 12 13 
do
	v1=$((X*10+20)) # perhaps try to align this with 2 things on each numa node, and hten perhaps numba node 1 is free? 
	v2=$((X*10+9+20))
	echo "$v1 $v2"
	#tmux new-session -d -s t$X "taskset -c $v1-$v2 python density_functional_theory.py -gdb 9 -numerror -sk $X -backend cpu -float32  -level 0 -plevel 0 -debug  |& tee tmp/test$X.txt "
	#tmux new-session -d -s t$X "taskset -c $v1-$v2 python density_functional_theory.py -C $X -sk -1  -backend ipu -float32  -level 0 -plevel 0 -debug  |& tee tmp/test$X.txt "
	#tmux new-session -d -s t$X "taskset -c $v1-$v2 python density_functional_theory.py -C $X -sk 0 1 2 3 4 5 6 7 8 9 10 13 14 15  -backend cpu -float32  -level 0 -plevel 0 -debug  |& tee tmp/test$X.txt "
	tmux new-session -d -s t$X "taskset -c $v1-$v2 python density_functional_theory.py -C $X -sk -1 -backend ipu -float32  -level 0 -plevel 0 -debug  |& tee tmp/test$X.txt "
done

#tmux new-session -d -s tcpu 'python density_functional_theory.py -C 10 -backend cpu -float32  -generate -save  -level 0 -plevel 0 '