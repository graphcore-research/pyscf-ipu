clear
rm gen.so 

cc cpu_int2e_sph.cpp -shared -fpic -o gen.so  -lpoplar -lpoputil -fpermissive 
echo "Done compiling"
#echo "Calling from python";

export XLA_IPU_PLATFORM_DEVICE_COUNT=1
export POPLAR_ENGINE_OPTIONS="{
\"autoReport.outputExecutionProfile\": \"true\",
\"autoReport.directory\": \"profs/\"
}"
export TF_POPLAR_FLAGS=--show_progress_bar=true
python direct.py $@
