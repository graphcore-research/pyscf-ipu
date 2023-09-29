clear
rm libcint.so 

echo "Compiling with C++"
g++ _libcint.c -shared -fpic  -fpermissive -o libcint.so -lpoplar -lpoputil 
echo "Done compiling. Calling C code from python. "

#POPLAR_ENGINE_OPTIONS="{ \"autoReport.outputExecutionProfile\": \"true\", \"autoReport.directory\": \"profs/\" }" JAX_IPU_USE_MODEL=1 progress_bar=true python libcint.py
TF_POPLAR_FLAGS=--executable_cache_path="_cache/" XLA_IPU_PLATFORM_DEVICE_COUNT=1  TF_POPLAR_FLAGS=--show_progress_bar=true python libcint.py
