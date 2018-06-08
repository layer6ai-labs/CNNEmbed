import numpy as np
import tensorflow as tf
from models.CNNEmbed import CNNEmbed
import nltk
import cPickle
import os
import dataset_handler
import pdb
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

VOCAB_SIZE = 483019
ZERO_IND = 483018

tknzr = nltk.tokenize.TweetTokenizer()
from train_GBW import perform_trec_exp

def load_model():
    '''
    Load the CNN model
    '''

    # Model parameters here.
    context_len = 5
    batch_size = 100
    num_filters = 900
    filter_size = 5
    num_layers = 8
    pos_words_num = 5
    neg_words_num = 10
    num_residual = 1
    k_max = 3
    max_doc_len = 45
    embed_dim = 300

    cnn_model = dict()

    doc2vec_graph = tf.Graph()
    with doc2vec_graph.as_default(), tf.device("/cpu:0"):
        indices_data_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, None])
        indices_target_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, pos_words_num + neg_words_num])

        embedding = tf.get_variable("embedding", [VOCAB_SIZE, embed_dim], dtype=tf.float32, trainable=True)
        inputs = tf.gather(embedding, indices_data_placeholder)
        inputs = tf.expand_dims(inputs, 3)
        inputs = tf.transpose(inputs, [0, 2, 1, 3])

        targets_embeds = tf.gather(embedding, indices_target_placeholder)
        targets_embeds = tf.expand_dims(targets_embeds, 3)
        targets_embeds = tf.transpose(targets_embeds, [0, 2, 1, 3])

        target_place_holder = tf.placeholder(tf.float32, [None, pos_words_num + neg_words_num])
        # Placeholder for training
        keep_prob_placeholder = tf.placeholder(dtype=tf.float32, name='dropout_rate')
        is_training_placeholder = tf.placeholder(dtype=tf.bool, name='training_boolean')

        # build model
        _docCNN = CNNEmbed(inputs, targets_embeds, target_place_holder, is_training_placeholder, keep_prob_placeholder,
                           max_doc_len, embed_dim, num_layers, num_filters, num_residual, k_max, filter_size, 0.)

        # input of the test (supervised learning) process
        model_output = tf.squeeze(_docCNN.res)

        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_docCNN = tf.Session(config=session_conf)
        saver = tf.train.Saver()

    # Restore the weights
    saver.restore(sess_docCNN, os.path.join('./latest_model_gbw', 'gbw_model_latest'))

    cnn_model['sess'] = sess_docCNN
    cnn_model['model'] = _docCNN
    cnn_model['placeholders'] = [indices_data_placeholder, is_training_placeholder, keep_prob_placeholder]
    cnn_model['model_output'] = model_output

    return cnn_model


def perform_exp(cnn_model, word_to_index, experiments):
    '''
    Perform the listed classification experiments. Modelled off of skip-thought.
    '''

    for exp in experiments:
        print('--------------------------------------------')
        if exp == 'TREC':
            perform_trec_exp(cnn_model['sess'], cnn_model['model_output'], cnn_model['placeholders'][0],
                             cnn_model['placeholders'][2], cnn_model['placeholders'][1], word_to_index)
        else:
            # Load the dataset and extract features
            z, features = dataset_handler.load_data(cnn_model, word_to_index, exp)

            scan = [2 ** t for t in range(0, 9, 1)]
            kf = KFold(n_splits=10, random_state=1234)
            scores = []
            for train, test in kf.split(features):
                # Split data
                X_train = features[train]
                y_train = z['labels'][train]
                X_test = features[test]
                y_test = z['labels'][test]

                scanscores = []
                for s in scan:

                    # Inner KFold
                    innerkf = KFold(n_splits=10, random_state=1234+1)
                    innerscores = []
                    for innertrain, innertest in innerkf.split(X_train):

                        # Split data
                        X_innertrain = X_train[innertrain]
                        y_innertrain = y_train[innertrain]
                        X_innertest = X_train[innertest]
                        y_innertest = y_train[innertest]

                        # Train classifier
                        clf = LogisticRegression(C=s)
                        clf.fit(X_innertrain, y_innertrain)
                        acc = clf.score(X_innertest, y_innertest)
                        innerscores.append(acc)

                    # Append mean score
                    scanscores.append(np.mean(innerscores))

                # Get the index of the best score
                s_ind = np.argmax(scanscores)
                s = scan[s_ind]
                print('Best value for C: {}'.format(s))

                # Train classifier
                clf = LogisticRegression(C=s)
                clf.fit(X_train, y_train)

                # Evaluate
                acc = clf.score(X_test, y_test)
                scores.append(acc)

            print('{} classification accuracy: {}'.format(exp, np.mean(scores)))


if __name__ == '__main__':

    cnn_model = load_model()
    with open('./gbw_cache/word_to_index.pkl', 'r') as f:
        word_to_index = cPickle.load(f)

    experiments = ['TREC', 'MR', 'CR', 'SUBJ', 'MPQA']

    perform_exp(cnn_model, word_to_index, experiments)
