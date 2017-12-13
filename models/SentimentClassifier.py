import tensorflow as tf


class SentimentClassifier(object):
    '''
    A Simple feedforward network to perform multi-class classification.
    '''

    def __init__(self, input_placeholder, labels_placeholder, embed_dim, batch_size, num_classes):
        '''
        Args:
            input_placeholder: (batch_size x embed_dim) Tensorflow placeholder to store the inputs.
            labels_placeholder: (batch_size) Tensorflow placeholder to store the labels.
            embed_dim (int): The dimensionality of the embedding/input
            batch_size (int): The batch size
            num_classes (int): Number of output classes
        '''

        self.input_placeholder = input_placeholder
        self.labels_one_hot = tf.one_hot(tf.cast(labels_placeholder, dtype=tf.int32), num_classes)
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.num_classes = num_classes

        # Build the classifier
        self.build_model()

    def build_model(self):
        '''
        Build the classifier model.
        '''

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

            hidden1_out = tf.nn.bias_add(tf.matmul(self.input_placeholder, weights), biases)
            hidden1_out = tf.nn.tanh(hidden1_out)

        with tf.variable_scope('classifier_output'):
            weights = tf.get_variable(
                name="weights",
                shape=[100, self.num_classes],
                initializer=tf.truncated_normal_initializer(stddev=1e-2, dtype=tf.float32),
                dtype=tf.float32)

            biases = tf.get_variable(
                name="biases",
                shape=[self.num_classes],
                initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                dtype=tf.float32)

            # Un-normalized logits
            self.logits = tf.nn.bias_add(tf.matmul(hidden1_out, weights), biases)

    def get_predictions(self):
        '''
        Return the classifier predictions
        '''

        return tf.argmax(self.logits, axis=1)

    def loss(self):
        '''
        Return the loss.
        '''

        classifier_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels_one_hot)
        classifier_loss = tf.reduce_mean(classifier_loss)
        return classifier_loss
