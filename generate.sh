# Install (takes 2-3 min) 
cd /notebooks/
pip install -q jax==0.3.16+ipu jaxlib==0.3.15+ipu.sdk320 -f https://graphcore-research.github.io/jax-experimental/wheels.html 
pip install -q git+https://github.com/graphcore-research/tessellate-ipu.git@main 
pip install -r requirements.txt 
pip install -r requirements_ipu.txt 
apt update 
apt -y install tmux 

# Start data generation (takes 2-3 min to compile)
tmux new-session -d -s 0 "XLA_IPU_PLATFORM_DEVICE_COUNT=1 python density_functional_theory.py -gdb 6 -randomSeed 0 -num_conformers 1000 -fname test -split 0 3 -threads 3 -threads_int 3  -generate -save -backend ipu -float32 -level 0 -plevel 0 -its 20 -skip_minao -choleskycpu -multv 2 -basis sto3g"
tmux new-session -d -s 1 "XLA_IPU_PLATFORM_DEVICE_COUNT=1 python density_functional_theory.py -gdb 6 -randomSeed 1 -num_conformers 1000 -fname test -split 0 3 -threads 3 -threads_int 3  -generate -save -backend ipu -float32 -level 0 -plevel 0 -its 20 -skip_minao -choleskycpu -multv 2 -basis sto3g"
tmux new-session -d -s 2 "XLA_IPU_PLATFORM_DEVICE_COUNT=1 python density_functional_theory.py -gdb 6 -randomSeed 2 -num_conformers 1000 -fname test -split 0 3 -threads 3 -threads_int 3  -generate -save -backend ipu -float32 -level 0 -plevel 0 -its 20 -skip_minao -choleskycpu -multv 2 -basis sto3g"
tmux new-session -d -s 3 "XLA_IPU_PLATFORM_DEVICE_COUNT=1 python density_functional_theory.py -gdb 6 -randomSeed 3 -num_conformers 1000 -fname test -split 0 3 -threads 3 -threads_int 3  -generate -save -backend ipu -float32 -level 0 -plevel 0 -its 20 -skip_minao -choleskycpu -multv 2 -basis sto3g"

tmux list-sessions

echo "Files are stored in data/generated/test/..."
echo "You can inspect the individual generation through 'tmux attach-session -t 0'. "
