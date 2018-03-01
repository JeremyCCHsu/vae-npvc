datadir="dataset/vcc2016/wav/"
datalink="https://datashare.is.ed.ac.uk/bitstream/handle/10283/2211/"
mkdir -p $datadir
wget "${datalink}license_text"
wget "${datalink}vcc2016_training.zip"
wget "${datalink}evaluation_all.zip"
unzip vcc2016_training.zip
unzip evaluation_all.zip
mv vcc2016_training "${datadir}Training Set"
mv evaluation_all "${datadir}Testing Set"
