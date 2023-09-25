clear
rm libcint.so 

echo "Compiling with C++"
g++ _libcint.c -shared -fpic  -fpermissive -w -o libcint.so -lpoplar -lpoputil 
echo "Done compiling. Calling C code from python. "


XLA_IPU_PLATFORM_DEVICE_COUNT=1  TF_POPLAR_FLAGS=--show_progress_bar=true python libcint.py
