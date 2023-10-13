clear
rm libcint.so 

g++ libcint.c -shared -fpic -o libcint.so -lpoplar -lpoputil -fpermissive
echo "Done compiling. Calling C code from python. "

POPLAR_ENGINE_OPTIONS="{
\"autoReport.all\": \"true\",
\"autoReport.directory\": \"profs/\"
}" XLA_IPU_PLATFORM_DEVICE_COUNT=1 TF_POPLAR_FLAGS=--show_progress_bar=true python libcint.py $@
