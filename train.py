import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from preprocess import *
from util import *
import pdb

if __name__ == '__main__':

    # command line tools for specifying the hyper-parameters only.
    parser = argparse.ArgumentParser(description='Train a CNN for embedding learning.')
    parser.add_argument('--context-len', type=int, help='The size of the minimum context.')
    parser.add_argument('--batch-size', type=int, help='Batch size.')
    parser.add_argument('--num-filters', type=int, help='Number of convolutional filters.')
    parser.add_argument('--num-layers', type=int, help='Number of layers, including the last fully-connected layer.')
    parser.add_argument('--num-positive-words', type=int, help='Number of next words to predict.')
    parser.add_argument('--num-negative-words', type=int, help='Number of negative samples.')
    parser.add_argument('--num-residual', type=int, default=-1, help='Number of layers to skip in residual connections.')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes in the classifier (2 or 5).')
    parser.add_argument('--dropout-keep-prob', type=float, default=0.8, help='The dropout keep prob.')
    parser.add_argument('--preprocessing', action='store_true', help='If redo the pre-processing. If set as False, the '
                                                                     'the program would try to load saved pre-processed'
                                                                     'files.')
    parser.add_argument('--cache-dir', type=str, default='./cache', help='The directory for saved pre-processed and'
                                                                           'embedding files')
    parser.add_argument('--dataset', type=str,required=True, help='Name of the dataset the model is training on.'
                                                                    'Either amazon or imdb.')
    parser.add_argument('--data-dir', type=str, default='/home/chundi/L6/Data/', help='Data directory.')
    parser.add_argument('--checkpoint-dir', type=str, default='./latest_model/', help='Checkpoints directory.')
    parser.add_argument('--model', type=str, default='CNN_pad',help='The model to use. Can be CNN_pad, CNN_pool or CNN_topk')

    args = parser.parse_args()

    context_len = args.context_len
    batch_size = args.batch_size
    num_filters = args.num_filters
    num_layers = args.num_layers
    pos_words_num = args.num_positive_words
    neg_words_num = args.num_negative_words
    num_residual = args.num_residual
    keep_prob = args.dropout_keep_prob
    data_dir = args.data_dir
    checkpoint_path = args.checkpoint_dir

    if args.dataset == 'imdb':
        max_doc_len = 400
        split_class = 7
        unlabeled_class = 0
    else: #argparse.dataset == 'amazon':
        max_doc_len = 200
        split_class = 2
        unlabeled_class = 2

    if args.model == 'CNN_pad':
        fixed_length = True
    else: #argparse.dataset == 'pool or topk':
        fixed_length = False


    cache_dir = args.cache_dir + '_' + args.dataset
    vector_up_fn = os.path.join(cache_dir, 'vector_up.npy')
    train_data_inds_fn = os.path.join(cache_dir, 'train_data_indices.npy')
    train_labels_fn = os.path.join(cache_dir, 'train_labels.npy')
    test_data_indices_fn = os.path.join(cache_dir, 'test_data_indices.npy')
    test_labels_fn = os.path.join(cache_dir, 'test_labels.npy')
    train_batches_fn = os.path.join(cache_dir, 'train_batches.npy')
    next_words_fn = os.path.join(cache_dir, 'next_words.npy')

    if not args.preprocessing:
        # Load the variables. This will generate an error if those files don't exist.
        vector_up = np.load(vector_up_fn)
        train_data_indices = np.load(train_data_inds_fn)
        train_labels = np.load(train_labels_fn)
        test_data_indices = np.load(test_data_indices_fn)
        test_labels = np.load(test_labels_fn)
    else:
        # Preprocess data
        if args.dataset == 'imdb':
            vector_up, train_data_indices, train_labels, test_data_indices, test_labels = get_data_imdb(data_dir, max_doc_len, fixed_length)
        else: #argparse.dataset == 'amazon':
            vector_up, train_data_indices, train_labels, test_data_indices, test_labels = get_data_amazon(data_dir, max_doc_len, fixed_length)
        np.save(vector_up_fn, vector_up)
        np.save(train_data_inds_fn, train_data_indices)
        np.save(train_labels_fn, train_labels)
        np.save(test_data_indices_fn, test_data_indices)
        np.save(test_labels_fn, test_labels)

    train_labels = train_labels.reshape([train_labels.shape[0]])
    test_labels = test_labels.reshape([test_labels.shape[0]])
    # Get the supervised test data
    if args.num_classes == 2:
        I = train_labels != unlabeled_class
        J = test_labels != unlabeled_class
        train_data_indices_sup = train_data_indices[I]
        test_data_indices_sup = test_data_indices[J]
        if fixed_length:
            train_data_indices_sup = pad_zeros(train_data_indices_sup, vector_up.shape[0] - 1, max_doc_len)
            test_data_indices_sup = pad_zeros(test_data_indices_sup, vector_up.shape[0] - 1, max_doc_len)
        train_labels_sup = train_labels[I]
        test_labels_sup = test_labels[J]
        train_labels_sup = train_labels_sup > split_class
        test_labels_sup = test_labels_sup > split_class
    elif args.num_classes == 5:
        if args.dataset == 'imdb':
            print("Imdb dataset only supports binary classification!")
            sys.exit()
        if fixed_length:
            train_data_indices_sup = pad_zeros(train_data_indices, vector_up.shape[0] - 1, max_doc_len)
            test_data_indices_sup = pad_zeros(test_data_indices, vector_up.shape[0] - 1, max_doc_len)
        else:
            train_data_indices_sup = np.copy(train_data_indices)
            test_data_indices_sup = np.copy(test_data_indices)
        train_labels_sup = np.copy(train_labels)
        test_labels_sup = np.copy(test_labels)
    else:
        print("Number of classes has to be 2 or 5!")
        sys.exit()
    pdb.set_trace()
    pass
    print "all of our inputs would follow NHWC _batch_height_width_channel_"
