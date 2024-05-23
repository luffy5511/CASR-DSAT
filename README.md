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
Download [MEAlign](https://pan.baidu.com/s/19lXZYWvs95eK8v2JLgEbXw). code: 1234
Download [M2I learning of CASR-DSAT](https://pan.baidu.com/s/19lXZYWvs95eK8v2JLgEbXw). code: 1234
Download [GM2M learning of CASR-DSAT](https://pan.baidu.com/s/19lXZYWvs95eK8v2JLgEbXw). code: 1234
# Pre-trained models of supervised comparison methods
Download [supervised comparison methods](https://pan.baidu.com/s/11b5XnLtvQcWmjZ70gpfZCA). code: 1234
# Pre-trained models of self-supervised comparison methods
Download [self-supervised comparison methods](https://pan.baidu.com/s/1X84_uM-S8RH6ltylQ3biAw). code: 1234
# Installation
python=3.7
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```
See requirements.txt for the installation of dependencies required
```
pip install -r requirements.txt
```
