#!/bin/bash
FLAGS="-N 1 -p sub64w --exclusive --constraint=bow"
if [ "$*" == "" ]
then
	salloc $FLAGS -t 48:0:0 
else
	sbatch $FLAGS $*
fi
