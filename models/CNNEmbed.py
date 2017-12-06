import tensorflow as tf
import numpy as np
import pdb


class CNNEmbed(object):
    '''
    Class for building a document embedding model using CNNs.
    '''

    def __init__(self, input_data, target_labels, keep_prob, max_doc_len=400, embed_dim=300,
                 num_layers=4, num_filters=900, residual_skip=2, k_max=0):
        '''
        Create a CNN for learning document embeddings.

        Args:
            input_data: (batch_size x embed_dim x doc_len x 1) tensor of word embeddings.
            target_embeddings: (batch_size x embed_dim x (num_pos_words + num_neg_words) x 1) tensor of target word
                embeddings.
            target_labels: (batch_size x (num_pos_words + num_neg_words)) tensor of target labels.
            keep_prob (placeholder): TF placeholder used for setting the keep prob during dropout
            max_doc_len (int): Length of document
            embed_dim (int): Dimensionality of the embedding
            num_layers (int): Number of layers, including the top-most fully-connected layer
            num_filters (int): Number of filters for each convolutional layer
            residual_skip (int): Number of layers to skip for res-net connections.
            k_max (int): The value of k when performing k-max pooling
        '''

        self.input_data = input_data
        self.target_labels = target_labels
        self.keep_prob = keep_prob
        self.max_word_num = max_doc_len
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.residual_skip = residual_skip
        self.k_max = k_max

        # Build the model.
        self.build_model()

    def build_model(self):
        '''
        Build the CNN model.
        '''

        prev_layer = self.input_data
        res_input = None
        std = np.sqrt(2. / (1 * 5 * self.num_filters))

        for i in range(self.num_layers - 1):
            with tf.variable_scope('conv_{}'.format(i)):
                if i == 0:
                    filter_height = self.embed_dim
                    filter_width = 3
                    in_chans = 1
                else:
                    filter_height = 1
                    filter_width = 5
                    in_chans = self.num_filters

                # Dropout layer
                prev_layer = tf.nn.dropout(prev_layer, keep_prob=self.keep_prob)

                std = np.sqrt(2. / (1 * 5 * self.num_filters))
                conv_w = self.conv_op(prev_layer, filter_width, filter_height, in_chans, 'w_' + str(i), std)
                conv_v = self.conv_op(prev_layer, filter_width, filter_height, in_chans, 'v_' + str(i), std)

                # Adding the gatting.
                gated_conv = tf.multiply(conv_w, tf.sigmoid(conv_v))

                # Residual connections
                if self.residual_skip and (i + 1) % self.residual_skip == 0 and res_input is not None:
                    gated_conv = (gated_conv + res_input) * tf.sqrt(0.5)

                if self.residual_skip and i == 0:
                    res_input = gated_conv
                elif self.residual_skip and (i + 1) % self.residual_skip == 0:
                    res_input = gated_conv

                prev_layer = gated_conv


        # Final fully connected block.
        with tf.variable_scope('fully_connected'):
            if self.k_max:
                # If we're doing k-max pooling
                output = tf.nn.top_k(tf.transpose(prev_layer, [0, 1, 3, 2]), self.k_max)[0]
                output = tf.reshape(output, [-1, self.k_max * self.num_filters])
                weights = tf.get_variable(name='weights', shape=[self.k_max * self.num_filters, self.embed_dim],
                                          dtype=tf.float32,
                                          initializer=tf.random_normal_initializer(0.0, std))
                biases = tf.get_variable(name='biases', shape=[self.embed_dim], dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.0))
                self.res = tf.nn.bias_add(tf.matmul(output, weights), biases)
            else:
                average_h = tf.squeeze(tf.reduce_max(prev_layer, axis=2), axis=1)
                weights = tf.get_variable(name='weights', shape=[self.num_filters, self.embed_dim], dtype=tf.float32,
                                          initializer=tf.random_normal_initializer(0.0, std))
                biases = tf.get_variable(name='biases', shape=[self.embed_dim], dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.0))
                self.res = tf.nn.bias_add(tf.matmul(average_h, weights), biases)

            self.res = tf.expand_dims(tf.expand_dims(self.res, 0), 0)
            self.res = tf.transpose(self.res, perm=[2, 1, 0, 3])

    def conv_op(self, fan_in, filter_width, filter_height, in_chans, name, std):
        '''
        Create a convolutional layer.

        Args:
            fan_in: Input tensor to the convoluational operation.
            filter_width (int): Width of the conv filter
            filter_height (int): height of the conv filter
            in_chans (int): Number of input channels
            name (str): Name to use for the tensor
            std (float): Standard deviation used to initialize the tensor values.

        Returns:
            An output tensor, after applying the convolution operation.
        '''

        paddings = [[0, 0], [0, 0], [filter_width / 2, filter_width / 2], [0, 0]]
        fan_in = tf.pad(fan_in, paddings, 'CONSTANT')

        kernel = tf.get_variable(
            name='weights_{}'.format(name),
            shape=[filter_height, filter_width, in_chans, self.num_filters],
            initializer=tf.random_normal_initializer(0., std),
            dtype=tf.float32)

        conv = tf.nn.conv2d(
            input=fan_in,
            filter=kernel,
            strides=[1, 1, 1, 1],
            padding='VALID',
            data_format='NHWC')

        biases = tf.get_variable(
            name='biases_{}'.format(name),
            shape=[self.num_filters],
            initializer=tf.constant_initializer(0.0, dtype=tf.float32),
            dtype=tf.float32)

        return tf.nn.bias_add(conv, biases)

    def loss(self):
        '''
        Return the sigmoid loss.
        '''

        scores = tf.squeeze(self.res)

        with tf.variable_scope('classifier_hidden'):
            weights = tf.get_variable(
                name="weights",
                shape=[self.embed_dim, 100],
                initializer=tf.truncated_normal_initializer(stddev=1e-2, dtype=tf.float32),
                dtype=tf.float32)

            biases = tf.get_variable(
                name="biases",
                shape=[100],
                initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                dtype=tf.float32)

            hidden1_out = tf.nn.bias_add(tf.matmul(scores, weights), biases)
            hidden1_out = tf.nn.tanh(hidden1_out)

        with tf.variable_scope('classifier_output'):
            weights = tf.get_variable(
                name="weights",
                shape=[100, 2],
                initializer=tf.truncated_normal_initializer(stddev=1e-2, dtype=tf.float32),
                dtype=tf.float32)

            biases = tf.get_variable(
                name="biases",
                shape=[2],
                initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                dtype=tf.float32)

            # Un-normalized logits
            self.logits = tf.nn.bias_add(tf.matmul(hidden1_out, weights), biases)
            self.target_one_hot = tf.one_hot(self.target_labels, 2)


        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.target_one_hot)
        return tf.reduce_mean(losses)

    def accuracy(self):
        '''
        Return the accuracy
        '''
        classifier_predictions = tf.argmax(self.logits, 1, name="predictions")
        correct_predictions = tf.equal(classifier_predictions, tf.argmax(self.target_one_hot, 1))
        accuracy_op = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        return accuracy_op
