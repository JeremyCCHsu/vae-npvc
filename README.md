# Voice Conversion from Non-parallel Corpora Using Variational Auto-encoder
Part of the code used in [our paper](https://arxiv.org/abs/1610.04019).  

This repo does not include the following modules due to authorship:  
 - feature extraction
 - F0 conversion module
 - waveform synthesis  

*The modules above are maintained by one of my colleagues.*

# Dependency
Tensorflow 1.1

## Feature
### Extraction
Starting from wav  
-> STRAIGHT spectrum (denoted by `sp`; frame rate: 5 ms, frame length: 20 ms)  
-> frame-wise energy normalization   
-> log (base 10)  

Note:
  1. Underour setting, the resulting values were in the range of about [-6.5, 0].  
  2. The energy was extracted as an independent feature which we did not modify.  
     (energy was unmodified during voice conversion)
  
### TF Record format
The `sp` was stored in raw binary format (4 bytes per value).  
Format:
```
[[i, s1, s2, ..., s513],
 [i, s1, s2, ..., s513],
 ...,
 [i, s1, s2, ..., s513]]
```
where `i` is the speaker index (0 - 9) and `s` is the `sp`.  

Note:
  - The speaker identity `i` was stored in `np.float32` but will be converted into `tf.int64` in the `VCC2016TFRManager` in `vcc2016io`. 
  - I shouldn't have stored the speaker identity per frame;
    it was just for implementation simplicity. 

# Usage
 - Specify the network architecture and the training hyper-parameters in `architecture.json`.
 - Run `python train.py`
 - Get the resulting models in `logdir/train/[date]`
 - Run `python validate --logdir [logdir] --target_id [integer] --file_pattern [glob pattern]`
 - Get the input, the reconstructed, and the converted spectra in the `logdir`.  

# About
The original code base was originally built in March, 2016.  
Tensorflow was in version 0.10 or earlier, so I decided to refactor my code and put it on this repo.
