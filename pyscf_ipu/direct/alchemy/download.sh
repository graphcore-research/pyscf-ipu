wget -O dev.zip https://alchemy.tencent.com/data/dev_v20190730.zip
wget -O valid.zip https://alchemy.tencent.com/data/valid_v20190730.zip
wget -O test.zip https://alchemy.tencent.com/data/test_v20190730.zip

unzip dev.zip 
unzip valid.zip 
unzip test.zip 

wget -O alchemy.zip https://alchemy.tencent.com/data/alchemy-v20191129.zip
unzip alchemy.zip 
mv Alchemy-v20191129/* . 
rmdir Alchemy-v20191129