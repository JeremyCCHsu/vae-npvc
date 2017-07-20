# Papers
1. [Voice Conversion from Non-parallel Corpora Using Variational Auto-encoder](https://arxiv.org/abs/1610.04019)  
2. [Voice Conversion from Unaligned Corpora using Variational Autoencoding Wasserstein Generative Adversarial Networks](https://arxiv.org/abs/1704.00849)
<br/>


# Dependency
Linux Ubuntu 16.04  
Python 3.5  

- Tensorflow-gpu 1.2.1
- Numpy
- Soundfile
- PyWorld
  - Cython
<br/>


# Setting up the environment
For example,  
```bash
conda create -n py35tf121 -y python=3.5
source activate py35tf121
pip install -U pip
pip install -r requirements.txt
```


### Note:
1. `soundfile` might require `sudo apt-get install`. 
2. You can use any virtual environment packages (e.g. `virtualenv`)
1. If your Tensorflow is the CPU version, you might have to replace all the `NCHW` ops in my code because Tensorflow-CPU only supports `NHWC` op and will report an error: `InvalidArgumentError (see above for traceback): Conv2DCustomBackpropInputOp only supports NHWC.`
2. I recommend installing Tensorflow from the link on their Github repo.  
    `pip install -U [*.whl link on the Github page]` 

<br/>


# Dataset
Voice Conversion Challenge 2016 (VCC2016): [download page](http://datashare.is.ed.ac.uk/handle/10283/2042)  
<br/>

# Model  
 - [x] Conditional VAE
 - [ ] Conditional VAE + WGAN

<br/>


# Usage
1. Run `bash download.sh` to prepare the VCC2016 dataset.  
2. Run `analyzer.py` to extract features and write features into binary files. (This takes a few minutes.)  
3. Run `build.py` to record some stats, such as spectral extrema and pitch.  
4. To train a VAE, for example, run
```bash
python main.py \
--model ConvVAE \
--trainer VAETrainer \
--architecture architecture-vae-vcc2016.json
```  
5. To convert the voice, run
```bash
python convert.py \
--src SF1 \
--trg TM3 \
--model ConvVAE \
--checkpoint logdir/train/0719-2303-34-2017/[model.ckpt-[id]] \
--file_pattern "./dataset/vcc2016/bin/Testing Set/{}/*.bin"
```
<br/>


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
<br/>



# Binary data format
The [WORLD vocdoer](https://github.com/mmorise/World) features and the speaker label are stored in binary format.  
Format:  
```
[[s1, s2, ..., s513, a1, ..., a513, f0, en, spk],
 [s1, s2, ..., s513, a1, ..., a513, f0, en, spk],
 ...,
 [s1, s2, ..., s513, a1, ..., a513, f0, en, spk]]
```
where   
`s_i` is spectral envelop magnitude (in log10) of the ith frequency bin,  
`a_i` is the corresponding "aperiodicity" feature,   
`f0` is the pitch (0 for unvoice frames),  
`en` is the energy,  
`spk` is the speaker index (0 - 9) and `s` is the `sp`.

Note:
  - The speaker identity `spk` was stored in `np.float32` but will be converted into `tf.int64` by the `reader` in `analysizer.py`.
  - I shouldn't have stored the speaker identity per frame;
    it was just for implementation simplicity. 

<br/>

# Modification Tips
1. Define a new model (and an accompanying trainer) and then specify the `--model` and `--trainer` of `main.py`.  
2. Tip: when creating a new trainer, overwrite `_optimize()` and the main loop in `train()`.

<br/>

# Difference from the original paper
1. WORLD vocoder is chosen in this repo over STRAIGHT because the former is open-sourced whereas the latter isn't.  
   I use [pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder), Python wrapper of the WORLD, in this repo.
2. Global variance post-filtering was not included in this repo.
<br/>


# About
The original code base was originally built in March, 2016.  
Tensorflow was in version 0.10 or earlier, so I decided to refactor my code and put it on this repo.


# TODO
 - [ ] util submodule (add to README)
 - [ ] Code architecture graph (UML)
 - [ ] license
 - [ ] Github