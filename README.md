# Enhanced Self-Supervised Multi-Image SuperResolution for Camera Array Images
# Pre-training
Download [Zurich RAW to RGB dataset](http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset).
```
python train_m2i.py
python train_gm2m.py
```
# Fine-tuning
Download [CASR real dataset](https://pan.baidu.com/s/175m1VXEwD5yo4PpngOktBw). code: 1234
```
python train_m2i_real.py
python train_gm2m_real.py
```
# Pre-trained model of CASR-DSAT
Download [MEAlign](http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset).
Download [M2I learning of CASR-DSAT](http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset).
Download [GM2M learning of CASR-DSAT](http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset).
# Pre-trained models of supervised comparison methods
Download [supervised comparison methods](http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset).
# Pre-trained models of self-supervised comparison methods
Download [self-supervised comparison methods](http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset).
# Installation
python=3.7
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```
See requirements.txt for the installation of dependencies required
```
pip install -r requirements.txt
```
