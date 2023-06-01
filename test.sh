for X in -2 -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
	v1=$((X*10+20)) 
	v2=$((X*10+9+20))
	echo "$v1 $v2"
	tmux new-session -d -s t$X "taskset -c $v1-$v2 python density_functional_theory.py -gdb 9 -enable64 -sk $X -backend cpu -float32  -level 0 -plevel 0 -debug  |& tee tmp/test$X.txt "
done

