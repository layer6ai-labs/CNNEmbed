import os
import numpy as np
from numpy.random import RandomState
from train_GBW import encode_text

def load_data(cnn_model, word_to_index, name, loc='/home/shunan/Code/SentEval/data/downstream', seed=1234):
    """
    Load one of MR, CR, SUBJ or MPQA
    """
    z = {}
    if name == 'MR':
        pos, neg = load_rt(loc=loc)
    elif name == 'SUBJ':
        pos, neg = load_subj(loc=loc)
    elif name == 'CR':
        pos, neg = load_cr(loc=loc)
    elif name == 'MPQA':
        pos, neg = load_mpqa(loc=loc)

    labels = compute_labels(pos, neg)
    text, labels = shuffle_data(pos+neg, labels, seed=seed)
    z['text'] = text
    z['labels'] = labels
    print 'Computing the encodings'
    features = []
    for sen in text:
        vec = encode_text(cnn_model['sess'], cnn_model['model_output'], cnn_model['placeholders'][0],
                               cnn_model['placeholders'][2], cnn_model['placeholders'][1], word_to_index, sen)
        features.append(vec)
    return z, np.array(features)


def load_rt(loc):
    """
    Load the MR dataset
    """
    pos, neg = [], []
    with open(os.path.join(loc, 'MR/rt-polarity.pos'), 'r') as f:
        for line in f:
            pos.append(line.decode('latin-1').strip())
    with open(os.path.join(loc, 'MR/rt-polarity.neg'), 'r') as f:
        for line in f:
            neg.append(line.decode('latin-1').strip())
    return pos, neg


def load_subj(loc):
    """
    Load the SUBJ dataset
    """
    pos, neg = [], []
    with open(os.path.join(loc, 'SUBJ/subj.objective'), 'r') as f:
        for line in f:
            pos.append(line.decode('latin-1').strip())
    with open(os.path.join(loc, 'SUBJ/subj.subjective'), 'r') as f:
        for line in f:
            neg.append(line.decode('latin-1').strip())
    return pos, neg


def load_cr(loc):
    """
    Load the CR dataset
    """
    pos, neg = [], []
    with open(os.path.join(loc, 'CR/custrev.pos'), 'r') as f:
        for line in f:
            text = line.strip()
            if len(text) > 0:
                pos.append(text)
    with open(os.path.join(loc, 'CR/custrev.neg'), 'r') as f:
        for line in f:
            text = line.strip()
            if len(text) > 0:
                neg.append(text)
    return pos, neg


def load_mpqa(loc):
    """
    Load the MPQA dataset
    """
    pos, neg = [], []
    with open(os.path.join(loc, 'MPQA/mpqa.pos'), 'r') as f:
        for line in f:
            text = line.strip()
            if len(text) > 0:
                pos.append(text)
    with open(os.path.join(loc, 'MPQA/mpqa.neg'), 'r') as f:
        for line in f:
            text = line.strip()
            if len(text) > 0:
                neg.append(text)
    return pos, neg


def compute_labels(pos, neg):
    """
    Construct list of labels
    """
    labels = np.zeros(len(pos) + len(neg))
    labels[:len(pos)] = 1.0
    labels[len(pos):] = 0.0
    return labels


def shuffle_data(X, L, seed=1234):
    """
    Shuffle the data
    """
    prng = RandomState(seed)
    inds = np.arange(len(X))
    prng.shuffle(inds)
    X = [X[i] for i in inds]
    L = L[inds]
    return (X, L)
