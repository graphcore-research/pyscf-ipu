#clear
rm gen.o
rm gen.so 

cc cpu_int2e_sph.cpp -shared -fpic -o gen.so  -lpoplar -lpoputil -fpermissive 
echo "Calling from python";

TF_POPLAR_FLAGS=--executable_cache_path=\"_cache/\"  python direct.py $@
#POPLAR_ENGINE_OPTIONS="{\"autoReport.all\": \"true\", \"autoReport.directory\": \"/a/scratch/alexm/research/popvision/poplar/\"}"  python direct.py $@
