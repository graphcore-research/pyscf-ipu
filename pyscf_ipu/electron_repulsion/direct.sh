clear
rm gen.so 

cc cpu_int2e_sph.cpp -shared -fpic -o gen.so  -lpoplar -lpoputil -fpermissive 
echo "Done compiling"
#echo "Calling from python";

TF_POPLAR_FLAGS=--show_progress_bar=true python direct.py $@
