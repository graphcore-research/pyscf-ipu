tmux new-session -d "taskset -c 0-29 python density_functional_theory.py $@ |& tee tmp/test0.txt" 
