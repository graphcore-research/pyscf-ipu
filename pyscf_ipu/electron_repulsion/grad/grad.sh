clear
rm gen.so 

cc grad.c -shared -fpic -o grad.so  
echo "Done compiling. Calling C code from python. "

python grad.py
