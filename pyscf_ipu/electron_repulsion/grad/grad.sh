clear
rm grad.so 

cc grad.c -shared -fpic -o grad.so -lpoplar -lpoputil -fpermissive
echo "Done compiling. Calling C code from python. "

python grad.py
