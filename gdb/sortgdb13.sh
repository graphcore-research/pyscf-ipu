#for X in $(seq 0 100)
#for X in $(seq 0 10)
#for X in $(seq 30 40)
#for X in $(seq 6 100)
for X in $(seq $1 $2)
do
  echo "$X"
  tmux new-session -d -s $X "taskset -c $X python sortgdb13.py $X"
done
