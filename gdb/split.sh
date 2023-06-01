input_file="13.smi"; output_prefix="/a/scratch/alexm/research/splitgdb13/gdb13_"; lines_per_file=10000000; 
split -l --numeric-suffixes=1 "${lines_per_file}" "${input_file}" "${output_prefix}"

