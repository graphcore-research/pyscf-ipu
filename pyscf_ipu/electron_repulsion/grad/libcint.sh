clear
rm libcint.so 

cc _libcint.c -shared -fpic -o libcint.so -lpoplar -lpoputil -fpermissive
echo "Done compiling. Calling C code from python. "

python libcint.py
