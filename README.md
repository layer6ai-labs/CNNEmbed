
<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logo_alt.png" width="180"></a>
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
This repository contains the full implementation, in Python, of the CNN-pad and CNN-pool models described in the paper
above. We also include scripts to perform training and evaluation. If you find this model useful in your research, 
please cite this paper:

```
@article{DBLP:journals/corr/abs-1711-04168,
  author    = {Chundi Liu and
               Shunan Zhao and
               Maksims Volkovs},
  title     = {Learning Document Embeddings With CNNs},
  journal   = {CoRR},
  volume    = {abs/1711.04168},
  year      = {2017},
  url       = {http://arxiv.org/abs/1711.04168},
  archivePrefix = {arXiv},
  eprint    = {1711.04168},
  timestamp = {Fri, 01 Dec 2017 14:22:24 +0100},
  biburl    = {http://dblp.org/rec/bib/journals/corr/abs-1711-04168},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
```


<a name="env"/>

## Environment
The python code is developed and tested on the following environment:
* python 2.7
* Intel i7-6800K
* 64GB RAM
* Nvidia GeForce GTX 1080 Ti
* CUDA 9 and CUDNN 8.0

Furthermore, we used the following libraries:
* tensorflow-gpu 1.4.0
* nltk 3.2.5
* scipy 1.0.0
* numpy 1.13.3


<a name="dataset"/>

## Dataset
To run the model, download the dataset from here and extract them to a `data` folder. The directory structure is as
follows:
```
data
  ├─ imdb_sentiment
  │   └─ imdb_sentiment.mat				
  ├─ amazon_food				
  │   ├─ train_data.pkl
  │   └─ test_data.pkl
  └─ word2vec
      ├─ GoogleNews-vectors-negative300.mat
      └─ dict.txt
```
Provide the path to the `data` folder as the argument to `--data-dir` when running `train.py`.

### Word2Vec
In our experiments, we initialize our word embeddings using pre-trained word2vec vectors. These can be downloaded
[here](https://code.google.com/archive/p/word2vec/).

### IMDB
The imdb dataset was obtained from [here](http://ai.stanford.edu/~amaas/data/sentiment/) and contains movies reviews
from the IMDB website, labelled by their sentiment score.

### Amazon Fine Food Reviews
The AFFR dataset was obtained from [here](https://www.kaggle.com/snap/amazon-fine-food-reviews). The original dataset is
highly inbalanced.


<a name="dataset"/>

## Training
Run the following command to reproduce the IMDB results:
```bash
python train.py 
```