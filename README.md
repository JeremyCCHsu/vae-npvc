# Papers
1. [Voice Conversion from Non-parallel Corpora Using Variational Auto-encoder](https://arxiv.org/abs/1610.04019)  
2. [Voice Conversion from Unaligned Corpora using Variational Autoencoding Wasserstein Generative Adversarial Networks](https://arxiv.org/abs/1704.00849)


# Installation
```bash
pip install -U pip
pip install -r requirements.txt
```

Note: `soundfile` might require `sudo apt-get install`


# Dependency
- Tensorflow 1.2.1
- Numpy
- Soundfile
- PyWorld
  - Cython


# Dataset
Voice Conversion Challenge 2016 (VCC2016)  
Download: http://datashare.is.ed.ac.uk/handle/10283/2042
Download all files


# Model
1. Conditional VAE
2. Conditional VAE + WGAN


# File/Folder
dataset
  vcc2016
    wav
    bin
      Training Set
      Testing Set
        SF1
        SF2
        ...
        TM3
etc
  xmax.npf
  xmin.npf
util (submodule)
model
logdir
architecture*.json

analyzer.py  
build.py  
trainer*.py
vcc2016_vae.py  
(validate.py)  
convert.py


# Usage
1. Run `analyzer.py` to extract features and write features into binary files.  
2. Run `build.py` to record some stats, such as spectral extrema and pitch


TODO:
Delete unnecessary files.



# Discrepancy
1. In the original paper, I used the STRAIGHT vocoder (which is not open-sourced).  
   However, in order to release this repo so that things can be reproduced,  
   I adopted the WORLD vocoder in this repo.
