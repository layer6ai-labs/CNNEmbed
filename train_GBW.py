import argparse
import tensorflow as tf
from preprocess import *
from util import *
from models.CNNEmbed import CNNEmbed
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
import os
import nltk
import codecs

RESTORE = True
CLASSIFICATION_DIR = '/home/shunan/Code/SentEval/data/downstream/TREC'
ZERO_IND = 483018

def encode_text(sess, model_output, indices_data_placeholder, keep_prob_placeholder, is_training_placeholder,
                word_to_index, text, doc_len=None):
    """
    Encode the text, which is just a sentence, as a vector using the CNN embedding model and return the output.
    """

    tknzr = nltk.tokenize.TweetTokenizer()
    tokens = nltk.word_tokenize(' '.join(tknzr.tokenize(text)))
    tokens = [word.lower() for word in tokens]
    line_tok = []
    for word in tokens:
        if word in word_to_index:
            line_tok.append(word_to_index[word])
        else:
            line_tok.append(word_to_index['<unk>'])

    if len(line_tok) < 5:
        # pad with zeros
        line_tok = [ZERO_IND for _ in range(5 - len(line_tok))] + line_tok

    if doc_len is not None:
        # pad or trucate to that length
        if len(line_tok) > doc_len:
            line_tok = line_tok[:doc_len]
        elif len(line_tok) < doc_len:
            line_tok = [ZERO_IND for _ in range(doc_len - len(line_tok))] + line_tok

    line_tok = np.array(line_tok)
    line_tok = np.reshape(line_tok, (1, len(line_tok)))

    # Feed it through model
    feed_dict = {indices_data_placeholder: line_tok, keep_prob_placeholder: 1., is_training_placeholder: False}
    encoding = sess.run([model_output], feed_dict)
    return encoding[0]


def perform_trec_exp(sess, model_output, indices_data_placeholder, keep_prob_placeholder,
                     is_training_placeholder, word_to_index):
    """
    Perform a classification experiments and output the results. For now, just perform classification on the TREC data,
    since there is a defined test/train split.
    """

    X_train, y_train, X_test, y_test = [], [], [], []
    tgt2idx = {'ABBR': 0, 'DESC': 1, 'ENTY': 2, 'HUM': 3, 'LOC': 4, 'NUM': 5}

    with codecs.open(os.path.join(CLASSIFICATION_DIR, 'train_5500.label'), 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip().split(':')
            y_train.append(tgt2idx[line[0]])
            X_train.append(encode_text(sess, model_output, indices_data_placeholder, keep_prob_placeholder,
                                       is_training_placeholder, word_to_index, line[1]))

    with codecs.open(os.path.join(CLASSIFICATION_DIR, 'TREC_10.label'), 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip().split(':')
            y_test.append(tgt2idx[line[0]])
            X_test.append(encode_text(sess, model_output, indices_data_placeholder, keep_prob_placeholder,
                                      is_training_placeholder, word_to_index, line[1]))

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_train, y_train = shuffle(X_train, y_train)

    # Fitting the logistic regression classifier.
    clf = LogisticRegression(C=128)
    clf.fit(X_train, y_train)

    print('TREC classification accuracy: {}'.format(str(clf.score(X_test, y_test))))


def training_pass(sess, train_op, data_inds, target_inds, batch_target, placeholders, keep_prob, is_training):
    """
    Do a training pass through a batch of the data.

    Args:
        sess: Tensorflow session.
        train_op: Tensorflow training operation.
        data_inds (numpy.ndarray): The training data, as an array of indices.
        target_inds (numpy.ndarray): The next words to predict
        batch_target (numpy.ndarray): The target labels
        placeholders (list): Tensorflow placeholders used for training
        keep_prob (float): The keep prob, used for dropout
        is_training (bool): Bool which is True if model is training, False if performing inference.

    Returns:
        None
    """

    indices_data_placeholder = placeholders[0]
    indices_target_placeholder = placeholders[1]
    target_place_holder = placeholders[2]
    kp_placeholder = placeholders[3]
    is_training_placeholder = placeholders[4]

    feed_dict = {indices_data_placeholder: data_inds, indices_target_placeholder: target_inds,
                 target_place_holder: batch_target, kp_placeholder: keep_prob, is_training_placeholder: is_training}
    sess.run([train_op], feed_dict)


def main(args):

    context_len = args.context_len
    batch_size = args.batch_size
    num_filters = args.num_filters
    filter_size = args.filter_size
    num_layers = args.num_layers
    pos_words_num = args.num_positive_words
    neg_words_num = args.num_negative_words
    num_residual = args.num_residual
    keep_prob = args.dropout_keep_prob
    l2_coeff = args.l2_coeff
    learning_rate = args.learning_rate
    data_dir = args.data_dir
    checkpoint_path = args.checkpoint_dir
    max_iter = args.max_iter
    k_max = args.top_k

    hyper_param_list = {'context_len': context_len, 'batch_size': batch_size, 'num_filters': num_filters,
                        'filter_size': filter_size, 'num_layers': num_layers, 'pos_words_num': pos_words_num,
                        'neg_words_num': neg_words_num, 'num_residual': num_residual, 'keep_prob': keep_prob,
                        'l2_coeff': l2_coeff}

    max_doc_len = 45
    embed_dim = 300
    vector_up = np.load(os.path.join(args.cache_dir, 'vector_up.npy'))
    with open(os.path.join(args.cache_dir, 'word_to_index.pkl'), 'r') as f:
        word_to_index = cPickle.load(f)

    indices_files = glob.glob(os.path.join(data_dir, 'gbw/tokenized/*'))

    ###########################################Embedding learning Graph#########################################
    doc2vec_graph = tf.Graph()
    with doc2vec_graph.as_default(), tf.device("/gpu:0"):
        indices_data_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, None])
        indices_target_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, pos_words_num + neg_words_num])

        embedding = tf.get_variable("embedding", [vector_up.shape[0], embed_dim], dtype=tf.float32, trainable=True)
        assign_embedding_op = tf.assign(embedding, vector_up)
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
                           max_doc_len, embed_dim, num_layers, num_filters, num_residual, k_max, filter_size, l2_coeff)

        global_step = tf.Variable(0, trainable=False)

        loss = _docCNN.loss()

        # input of the test (supervised learning) process
        model_output = tf.squeeze(_docCNN.res)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # setting the learning rate
        with tf.control_dependencies(update_ops):
            learning_rate_t = tf.train.exponential_decay(learning_rate, global_step, 1, 0.99)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_t)
            grads_and_vars = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(grads_and_vars)

        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_docCNN = tf.Session(config=session_conf)
        train_init_op_docCNN = tf.global_variables_initializer()
        saver = tf.train.Saver()

    ###########################################Training######################################
    # Initializing the variables.

    if RESTORE:
        saver.restore(sess_docCNN, os.path.join(checkpoint_path, 'gbw_model-0'))
    else:
        sess_docCNN.run(train_init_op_docCNN)
        sess_docCNN.run(assign_embedding_op)

    batch_target = np.hstack((np.full((batch_size, pos_words_num), 1), np.full((batch_size, neg_words_num), 0)))
    doc_lengths = [15, 24, 32, 41, 47]
    super_batch_size = 10000  # use the same doc len in a super batch
    placeholders = [indices_data_placeholder, indices_target_placeholder, target_place_holder,
                    keep_prob_placeholder, is_training_placeholder]

    iter = 1
    while iter < max_iter:
        file_num = 0
        for tokenized_file in indices_files:
            train_indices = np.load(tokenized_file)
            np.random.shuffle(train_indices)

            # we randomize the document lengths, so the model sees both long and short docs/sentences.
            doc_len = np.random.choice(doc_lengths)
            ind1 = 0
            ind2 = super_batch_size
            while ind1 < len(train_indices):
                curr_train_inds = train_indices[ind1:ind2]
                batch_generator = BatchGenerator(curr_train_inds, pos_words_num, neg_words_num, doc_len, context_len,
                                                 vector_up.shape[0] - 1, batch_size, vector_up.shape[0] - 1)
                data_size = batch_generator.get_data_size()
                num_batches = data_size / batch_size
                batch_generator.generate_training_batches()

                for i in range(num_batches):
                    ret_val = batch_generator.get_data()
                    data_inds, target_inds = ret_val
                    training_pass(sess_docCNN, train_op, data_inds, target_inds, batch_target, placeholders, keep_prob,
                                  True)

                doc_len = np.random.choice(doc_lengths)
                ind1 += super_batch_size
                ind2 += super_batch_size

            # Finished one of the files
            feed_dict = {indices_data_placeholder: data_inds, indices_target_placeholder: target_inds,
                         target_place_holder: batch_target, keep_prob_placeholder: 1., is_training_placeholder: False}
            loss_out = sess_docCNN.run([loss], feed_dict)
            print('Epoch: {}, file: {}, loss: {}'.format(iter, file_num + 1, loss_out))
            print('-----------------------------------------------')
            if (file_num + 1) % 25 == 0:
                print('Performing classification experiment')
                perform_trec_exp(sess_docCNN, model_output, indices_data_placeholder,
                                 keep_prob_placeholder, is_training_placeholder, word_to_index)

            file_num += 1
            saver.save(sess_docCNN, os.path.join(checkpoint_path, 'gbw_model_latest'))

        # Finished one epoch.
        print('Completed one epoch.')
        saver.save(sess_docCNN, os.path.join(checkpoint_path, 'gbw_model'), global_step=iter)
        iter += 1


if __name__ == '__main__':

    # command line tools for specifying the hyper-parameters and other training options.
    parser = argparse.ArgumentParser(description='Train a CNN for embedding learning.')
    parser.add_argument('--context-len', default=5, type=int, help='The size of the minimum context.')
    parser.add_argument('--batch-size', default=100, type=int, help='Batch size.')
    parser.add_argument('--num-filters', type=int, default=900, help='Number of convolutional filters.')
    parser.add_argument('--filter-size', type=int, default=5, help='The size of the convolutional filters.')
    parser.add_argument('--num-layers', type=int, default=8,
                        help='Number of layers, including the last fully-connected layer.')
    parser.add_argument('--num-positive-words', type=int, default=5, help='Number of next words to predict.')
    parser.add_argument('--num-negative-words', type=int, default=10, help='Number of negative samples.')
    parser.add_argument('--num-residual', type=int, default=1, help='Number of layers to skip in residual connections.')
    parser.add_argument('--dropout-keep-prob', type=float, default=0.8, help='The dropout keep prob.')
    parser.add_argument('--l2-coeff', type=float, default=0., help='The weight decay coefficient (l2).')
    parser.add_argument('--cache-dir', type=str, default='./gbw_cache',
                        help='The directory containing the saved pre-processed and embedding files')
    parser.add_argument('--data-dir', type=str, default='/home/shunan/Data/', help='Directory containing the data.')
    parser.add_argument('--checkpoint-dir', type=str, default='./latest_model_gbw/', help='Checkpoints directory.')
    parser.add_argument('--learning-rate', type=float, default=0.0003, help='The learning rate.')
    parser.add_argument('--top-k', type=int, default=3, help='The value of k when performing k-max pooling')
    parser.add_argument('--max-iter', type=int, default=10, help='The maximum number of training iterations.')

    args = parser.parse_args()
    main(args)
