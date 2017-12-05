import os
import cPickle
import nltk
import struct
import numpy as np
from scipy.io import loadmat


def tokenize_sentence(data, data_type, word_to_index, max_doc_len, fixed_length):
    """
    Convert data, an array containing IMDB sentiment data, into a list of indices into the word2vec matrix.

    Args:
        data (numpy.ndarray): An array containing the IMDB sentiment data, as unicode strings.
        data_type (str): Data to use, either 'amazon' or 'imdb'
        word_to_index (dict): Dict that maps words to their index in the word2vec matrix
        max_doc_len (int): Maximum length of input, if using CNN-pad
        fixed_length (bool): True if using CNN-pad

    Returns:
        tokenized (list): list of lists, where each list is a document from the IMDB data converted to their word2vec
            indices
    """

    tokenized = []
    tknzr = nltk.tokenize.TweetTokenizer()

    for sentence in data:
        if data_type == 'imdb':
            sentence = sentence[0][0]
            tokens = nltk.word_tokenize(sentence)
        else:
            tokens = nltk.word_tokenize(' '.join(tknzr.tokenize(sentence)))

        tokenized_sentence = [word.lower() for word in tokens]
        tokenized_sentence = [word for word in tokenized_sentence if word in word_to_index]
        index_list = [word_to_index[word] for word in tokenized_sentence]
        length = len(index_list)
        if fixed_length and length > max_doc_len:
            index_list = index_list[:max_doc_len]

        tokenized.append(index_list)

    return tokenized


def load_word2vec(data_path):
    """
    Load the pre-trained word2vec vectors and return them, along with a mapping from words to their index in word2vec.

    Args:
        data_path (str): Path to the directory containing word2vec

    Returns:
        word_vectors (numpy.ndarray): Array of pre-trained word2vec vectors.
        word_to_index (dict): Mapping from words in word2vec to their index in word_vectors.
    """

    word2vec_f = open(os.path.join(data_path, 'word2vec/GoogleNews-vectors-negative300.bin'), 'rb')

    c = None
    # read the header
    header = ''
    while c != '\n':
        c = word2vec_f.read(1)
        header += c

    num_vectors, vector_len = (int(x) for x in header.split())
    word_vectors = np.zeros((num_vectors, vector_len))
    word_to_index = dict()
    float_size = 4  # 32bit float

    for n in range(num_vectors):
        word = []
        c = word2vec_f.read(1)
        while c != ' ':
            word.append(c)
            c = word2vec_f.read(1)
        word = ''.join(word).strip()

        binary_vector = word2vec_f.read(float_size * vector_len)
        vec = [struct.unpack_from('f', binary_vector, i)[0] for i in xrange(0, len(binary_vector), float_size)]
        word_vectors[n, :] = np.array(vec)
        word_to_index[word] = n

    return word_vectors, word_to_index


def get_data_imdb(data_path, max_doc_len, fixed_length=True):
    """
    Return the IMDB test and training data as a list of lists of indices.

    Args:
        data_path (str): Path to the directory containing the data.
        max_doc_len (int): Maximum length of input, if using CNN-pad
        fixed_length (bool): True if using CNN-pad

    Returns:
        input_embeddings (numpy.ndarray): Input word embeddings to the CNN
        train_data_indices (list): list of list of indices to word2vec, for training
        train_labels (numpy.ndarray): 1D array of labels, for training
        test_data_indices (list): list of list of indices to word2vec, for testing
        test_labels (numpy.ndarray): 1D array of labels, for training
    """

    # Read pre-trained word2vec vectors and dictionary
    word_vectors, word_to_index = load_word2vec(data_path)

    # Read train test data and label
    temp = loadmat(os.path.join(data_path, 'imdb_sentiment/imdb_sentiment.mat'))
    train_data = temp['train_data']
    test_data = temp['test_data']
    train_labels = temp['train_labels']
    test_labels = temp['test_labels']

    print('Tokenizing data and converting to indices.')
    train_data_indices = tokenize_sentence(train_data, 'imdb', word_to_index, max_doc_len, fixed_length)
    test_data_indices = tokenize_sentence(test_data, 'imdb', word_to_index, max_doc_len, fixed_length)

    # Create the unique word dict used by the model
    flatten_train = [item for sublist in train_data_indices for item in sublist]
    train_data_indices_unique = list(set(flatten_train))

    flatten_test = [item for sublist in test_data_indices for item in sublist]
    test_data_indices_unique = list(set(flatten_test))

    train_data_indices_unique.extend(test_data_indices_unique)
    all_unique_indices = list(set(train_data_indices_unique))

    reverse_index = {}
    for i in range(len(all_unique_indices)):
        reverse_index[all_unique_indices[i]] = i

    input_embeddings = word_vectors[all_unique_indices]
    # add an empty to vector and reverse vector
    input_embeddings = np.vstack([input_embeddings, np.zeros(input_embeddings.shape[1])])
    reverse_index[-1] = input_embeddings.shape[0] - 1
    print('Number of unique words in this dataset is {}'.format(len(input_embeddings)))

    for i in range(len(train_data_indices)):
        for j in range(len(train_data_indices[i])):
            train_data_indices[i][j] = reverse_index[train_data_indices[i][j]]
    for i in range(len(test_data_indices)):
        for j in range(len(test_data_indices[i])):
            test_data_indices[i][j] = reverse_index[test_data_indices[i][j]]

    # Convert list to np array
    train_data_indices = np.array(train_data_indices)
    train_labels = np.array(train_labels)
    test_data_indices = np.array(test_data_indices)
    test_labels = np.array(test_labels)

    return input_embeddings, train_data_indices, train_labels, test_data_indices, test_labels


def get_data_amazon(data_path, max_doc_len, fixed_length=True):
    """
    Return the Amazon Fine Food Reviews test and training data as a list of lists of indices.

    Args:
        data_path (str): Path to the directory containing the data.
        max_doc_len (int): Maximum length of input, if using CNN-pad
        fixed_length (bool): True if using CNN-pad

    Returns:
        input_embeddings (numpy.ndarray): Input word embeddings to the CNN
        train_data_indices (list): list of list of indices to word2vec, for training
        train_labels (numpy.ndarray): 1D array of labels, for training
        test_data_indices (list): list of list of indices to word2vec, for testing
        test_labels (numpy.ndarray): 1D array of labels, for training
    """

    # Read pre-trained word2vec vectors and dictionary
    word_vectors, word_to_index = load_word2vec(data_path)

    with open(os.path.join(data_path, 'amazon_food/train_data.pkl')) as train_data_fn:
        train_data = cPickle.load(train_data_fn)

    with open(os.path.join(data_path, 'amazon_food/test_data.pkl')) as test_data_fn:
        test_data = cPickle.load(test_data_fn)

    train_text = train_data[0]
    train_score = train_data[1]
    test_text = test_data[0]
    test_score = test_data[1]
    all_text = train_text + test_text

    data_indices = tokenize_sentence(all_text, 'amazon', word_to_index, max_doc_len, fixed_length)
    # Create the unique word dict used by the model
    flatten_data = [item for sublist in data_indices for item in sublist]
    all_unique_indices = list(set(flatten_data))

    reverse_index = {}
    for i in range(len(all_unique_indices)):
        reverse_index[all_unique_indices[i]] = i

    input_embeddings = word_vectors[all_unique_indices]
    # add an empty to vector and reverse vector
    input_embeddings = np.vstack([input_embeddings, np.zeros([input_embeddings.shape[1]])])
    reverse_index[-1] = input_embeddings.shape[0] - 1
    print('Number of unique words in this dataset is {}'.format(len(input_embeddings)))

    # Convert index from whole vocabulary to local vocabulary
    for i in range(len(data_indices)):
        for j in range(len(data_indices[i])):
            data_indices[i][j] = reverse_index[data_indices[i][j]]
    data_indices = np.array(data_indices)

    train_data_indices = data_indices[:80000]
    train_labels = np.array(train_score)
    train_labels -= 1

    test_data_indices = data_indices[80000:]
    test_labels = np.array(test_score)
    test_labels -= 1

    return input_embeddings, train_data_indices, train_labels, test_data_indices, test_labels
