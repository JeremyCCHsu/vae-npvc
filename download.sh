mkdir dataset
cd dataset
mkdir wav
wget http://datashare.is.ed.ac.uk/download/10283/2042/SUPERSEDED_-_The_Voice_Conversion_Challenge_2016.zip
unzip SUPERSEDED_-_The_Voice_Conversion_Challenge_2016.zip -d vcc2016
cd vcc2016
unzip vcc2016_training.zip
unzip evaluation_all.zip
mv vcc2016_training "../wav/Training Set"
mv evaluation_all "../wav/Testing Set"
cd ../..
