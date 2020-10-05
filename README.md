!!!
<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/CNNEmbed/blob/prep/logos/logo_alt.png" width="180"></a>
</p>

# CNNEmbed
[Tensorflow](https://www.tensorflow.org/) implementation of
[Learning Document Embeddings With CNNs](https://arxiv.org/abs/1711.04168).

Authors: [Shunan Zhao](http://www.cs.toronto.edu/~szhao/), [Chundi Liu](https://ca.linkedin.com/in/chundiliu), 
[Maksims Volkovs](http://www.cs.toronto.edu/~mvolkovs)

## Table of Contents  
0. [Introduction](#intro)  
1. [Environment](#env)
2. [Dataset](#dataset)
2. [Training](#training)

<a name="intro"/>

## Introduction
This repository contains the full implementation, in Python, of the CNN document embedding models described in the paper
above. We also include scripts to perform training and evaluation. If you find this model useful in your research, 
please cite this paper:

```
@article{liu2017embed
  author    = {Chundi Liu and
               Shunan Zhao and
               Maksims Volkovs},
  title     = {Learning Document Embeddings With CNNs},
  journal   = {arXiv:1711.04168},
  year      = {2017}
}
```


<a name="env"/>

## Environment
The python code is developed and tested on the following environment:
* python 2.7
* Intel i7-6800K
* 64GB RAM
* Nvidia GeForce GTX 1080 Ti
* CUDA 8.0 and CUDNN 8.0

Furthermore, we used the following libraries:
* tensorflow-gpu 1.4.0
* nltk 3.2.5
* scipy 1.0.0
* numpy 1.13.3


<a name="dataset"/>

## Dataset
To run the model, download the dataset from [here](https://s3.amazonaws.com/public.layer6.ai/CNNEmbed/CNNEmbedData.tar.gz)
and extract them to a directory, refered to as `$DATA_DIR`. You should structure your data directory as follows:
```
data
  ├─ imdb_sentiment
  │   └─ imdb_sentiment.mat				
  ├─ amazon_food				
  │   ├─ amazon_train_data.pkl
  │   └─ amazon_test_data.pkl
  └─ word2vec
      └─ GoogleNews-vectors-negative300.bin
```
Provide `$DATA_DIR` as the argument to `--data-dir` when running `train.py`.

### Word2Vec
In our experiments, we initialize our word embeddings using pre-trained word2vec vectors. These can be downloaded
[here](https://code.google.com/archive/p/word2vec/).

### IMDB
The imdb dataset was obtained from [here](http://ai.stanford.edu/~amaas/data/sentiment/) and contains movies reviews
from the IMDB website, labelled by their sentiment score.

### Amazon Fine Food Reviews
The AFFR dataset was obtained from [here](https://www.kaggle.com/snap/amazon-fine-food-reviews). The original dataset is
highly unbalanced and contains many duplicates. As such, we removed all duplicates and balanced the classes, see our paper for full description of this dataset.

### Data Preparation
```bash
wget https://s3.amazonaws.com/public.layer6.ai/CNNEmbed/CNNEmbedData.tar.gz -O /tmp/CNNEmbedData.tar.gz
cd /tmp/
tar -zxvf CNNEmbedData.tar.gz
mv CNNEmbedData/* $DATA_DIR
```
Download the pre-trained word2vec embeddings [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing). 
Uncompress it and copy the binary file to the data directory.

```bash
cp GoogleNews-vectors-negative300.bin $DATA_DIR/word2vec/
```


<a name="training"/>

## Training

Because the pre-processing takes a long time, we store the pre-processed files in a cache directory, which you will need
to create and provide to the `--cache-dir` argument. You will also need to create a directory to store the Tensorflow
models and provide to the `--checkpoint-dir` argument. By default, they are set to be `./cache` and `./latest_model`.

Run the following command to reproduce the IMDB results:
```bash
python train.py --context-len=10 --batch-size=100 --num-filters=900 --num-layers=4 --num-positive-words=10 \
--num-negative-words=50 --num-residual=2 --num-classes=2 --dataset=imdb --model=CNN_topk --top-k=3 --max-iter=100 \
--data-dir=$DATA_DIR --preprocessing 
```
Notes:
* By default, both document embedding learning and classifier happen on single GPU.
* On our environment (described above), after 30 epochs (approximately 6 hours), the classifier gets 90% accuracy on the IMDB test set.
* If `train.py` has been run once and cached data are in the `--cache-dir`, removing `--preprocessing` parameter from train command would make it much faster. However, you **HAVE TO** redo the preprocessing whenever you want to change to a different dataset or model architecture.

## IMDB Results

Training loss and validation classification accuracy curves for the IMDB dataset. We train the classifier every 10 epochs and record the best accuracy (upper bound).

<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/CNNEmbed/blob/prep/logos/training_loss.png" width="70%"></a>
</p>

<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/CNNEmbed/blob/prep/logos/validation_classification_acc.png" width="70%"></a>
</p>
