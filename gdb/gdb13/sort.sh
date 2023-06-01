#for X in $(seq 0 100)
#for X in $(seq 0 10)
#for X in $(seq 30 40)
#for X in $(seq 6 100)
for X in 0 1 2 3 4 5 6 7 8 9 #$(seq $1 $2) 
do
  n=$(($X*10+$1))
  echo "$n"
  tmux new-session -d -s $n "taskset -c $n python sort.py $n"
done
