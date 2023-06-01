
AR=(134 136 137 138 140 142 144 146 148 150 152 154 158 160 162 165 170 172 175)
for X in -2 -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
	v1=$((X*10+20)) 
	v2=$((X*10+9+20))
	
	val=${AR[$X+2]} 
	echo "$v1 $v2 $val"

	tmux new-session -d -s t$X "taskset -c $v1-$v2 python density_functional_theory.py -gdb 9 -id $val -backend ipu -float32  -level 0 -plevel 0 -debug  |& tee tmp/$1test$X.txt "

done

#tmux new-session -d -s tcpu 'python density_functional_theory.py -C 10 -backend cpu -float32  -generate -save  -level 0 -plevel 0 '
