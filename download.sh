mkdir -p dataset/vcc2016/wav
cd dataset
wget "http://datashare.is.ed.ac.uk/download/10283/2042/SUPERSEDED_-_The_Voice_Conversion_Challenge_2016.zip"
unzip SUPERSEDED_-_The_Voice_Conversion_Challenge_2016.zip vcc2016_training.zip evaluation_all.zip
unzip vcc2016_training.zip
mv vcc2016_training "./vcc2016/wav/Training Set"
unzip evaluation_all.zip -d "vcc2016/wav/Testing Set"
rm evaluation_all.zip vcc2016_training.zip
cd ..
