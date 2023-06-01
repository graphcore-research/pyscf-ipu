for X in 1 2 3 4 5 6 7 8 9 10 11 12 13 
do
	v1=$((X*10+20)) # perhaps try to align this with 2 things on each numa node, and hten perhaps numba node 1 is free? 
	v2=$((X*10+9+20))
	echo "$v1 $v2"
	#tmux new-session -d -s t$X "taskset -c $v1-$v2 python density_functional_theory.py -C $X -sk -1 -backend ipu -ipumult -float32  -level 0 -plevel 0 -debug  |& tee tmp/$1test$X.txt "
	tmux new-session -d -s t$X "taskset -c $v1-$v2 python density_functional_theory.py -C $X -sk -1 -backend ipu -float32  -level 0 -plevel 0 -debug  |& tee tmp/$1test$X.txt "
done

