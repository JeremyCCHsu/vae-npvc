A modified version of our

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
Voice Conversion Challenge 2016 (VCC2016): [download page](http://datashare.is.ed.ac.uk/handle/10283/2042)
You can also download all the files by 
```bash
bash ./download.sh
```

# Model
1. Conditional VAE
2. Conditional VAE + WGAN


# File/Folder
```
dataset
  vcc2016
    bin
    wav
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

analyzer.py    (feature extraction)
build.py       (stats collecting)
trainer*.py
main.py        (main script)
(validate.py)  (output converted spectrogram) 
convert.py     (conversion)
```

# Usage
1. Run `analyzer.py` to extract features and write features into binary files.  
2. Run `build.py` to record some stats, such as spectral extrema and pitch
3. To train a VAE, for example, run
```bash
python main.py 
--model ConvVAE 
--trainer VAETrainer
--architecture architecture-vae-vcc2016.json
```  
4. To convert the voice, run
```bash
python convert.py 
--checkpoint logdir/train/0719-2303-34-2017/model.ckpt-197324 
--src SF1 
--trg TM3 
--file_pattern ./dataset/vcc2016/bin/Testing Set/{}/*001.bin
```


# Modification Tips
1. Define a new model (and an accompanying trainer) and then specify the `--model` and `--trainer` of `main.py`.  
2. Tip: when creating a new trainer, overwrite `_optimize()` and the main loop in `train()`.


# Difference from the original paper
1. WORLD vocoder is chosen in this repo over STRAIGHT because the former is open-sourced whereas the latter isn't.
2. Global variance post-filtering was not included in this repo.


## TODO:
- [x] Delete unnecessary files.
