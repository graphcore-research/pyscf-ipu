# use tmux to start 16 threads and then generate stuff on all of them! 
python density_functional_theory.py -C 6 -backend ipu -float32  -generate -save -level 1 -plevel 1
