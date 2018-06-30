import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from helper import parse_tfrecord


# ======================================================================================================================
# Wrap layers
# ======================================================================================================================
def conv_relu(inputs, kernel_shape, scope, reuse=False):
    bias_shape = [kernel_shape[-1]]
    with tf.variable_scope(scope, reuse=reuse):
        weights = tf.get_variable('weights', kernel_shape, initializer=tf.glorot_uniform_initializer())
        biases = tf.get_variable('biases', bias_shape, initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(inputs, weights, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)


def max_pool(inputs, k=2):
    return tf.nn.max_pool(inputs, [1, k, k, 1], [1, k, k, 1], padding='SAME')


def dense_relu(inputs, kernel_shape, scope, reuse=False):
    bias_shape = [kernel_shape[-1]]
    with tf.variable_scope(scope, reuse=reuse):
        weights = tf.get_variable('weights', kernel_shape, initializer=tf.glorot_uniform_initializer())
        biases = tf.get_variable('biases', bias_shape, initializer=tf.constant_initializer(0.0))
    return tf.nn.relu(tf.matmul(inputs, weights) + biases)


def dense(inputs, kernel_shape, scope, reuse=False):
    bias_shape = [kernel_shape[-1]]
    with tf.variable_scope(scope, reuse=reuse):
        weights = tf.get_variable('weights', kernel_shape, initializer=tf.glorot_uniform_initializer())
        biases = tf.get_variable('biases', bias_shape, initializer=tf.constant_initializer(0.0))
    return tf.matmul(inputs, weights) + biases


# ======================================================================================================================
# Define cnn mnist model
# ======================================================================================================================
class CNNModel(object):
    def __init__(self):
        # inputs
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.inputs = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='mnist_input')
        self.labels = tf.placeholder(tf.int32, shape=[None], name='mnist_label')

        # preprocess label
        self.onehot_labels = tf.one_hot(indices=self.labels, depth=10)

        # Convolutional Layer #1
        # [batch_size, 28, 28, 1] => [batch_size, 14, 14, 32]
        self.conv1 = conv_relu(self.inputs, kernel_shape=[5, 5, 1, 32], scope='conv1')
        self.pool1 = max_pool(self.conv1, k=2)

        # Convolutional Layer #2
        # [batch_size, 14, 14, 32] => [batch_size, 7, 7, 64]
        self.conv2 = conv_relu(self.pool1, kernel_shape=[5, 5, 32, 64], scope='conv2')
        self.pool2 = max_pool(self.conv2, k=2)

        # Flatten tensor into a batch of vectors
        # [batch_size, 7, 7, 64] => [batch_size, 7 * 7 * 64]
        self.flat3 = tf.reshape(self.pool2, shape=[-1, 7 * 7 * 64])

        # Dense Layer with dropout
        # [batch_size, 7 * 7 * 64] => [batch_size, 1024]
        self.dense4 = dense_relu(self.flat3, kernel_shape=[7 * 7 * 64, 1024], scope='dense4')
        self.dropout4 = tf.nn.dropout(self.dense4, keep_prob=self.keep_prob)

        # Logits layer
        # [batch_size, 1024] => [batch_size, 10]
        self.logits = dense(self.dropout4, kernel_shape=[1024, 10], scope='logits')

        # loss
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.onehot_labels, logits=self.logits)

        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        self.train_opt = optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())

        # model outputs
        self.pred = tf.cast(tf.argmax(self.logits, axis=1), dtype=tf.int32)
        self.correct_prediction = tf.cast(tf.equal(self.labels, self.pred), dtype=tf.float32)
        self.acc = tf.reduce_mean(self.correct_prediction)
        self.probs = tf.nn.softmax(self.logits)
        return


# ======================================================================================================================
# train with plain numpy array
# ======================================================================================================================
def train():
    # hyper parameters
    batch_size = 100
    epochs = 20

    # load mnist data
    mnist = input_data.read_data_sets('./data/mnist')
    test_images = np.reshape(mnist.test.images, newshape=[-1, 28, 28, 1])
    test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # create model
    model = CNNModel()

    # start training
    with tf.Session() as sess:
        # run initializer ops
        sess.run(tf.global_variables_initializer())

        # for each epoch
        for e in range(1, epochs + 1):
            for ii in range(mnist.train.num_examples // batch_size):
                # get data
                train_x, train_y = mnist.train.next_batch(batch_size)
                train_x = np.reshape(train_x, newshape=[-1, 28, 28, 1])

                # run train operation
                _ = sess.run(model.train_opt, feed_dict={
                    model.inputs: train_x,
                    model.labels: train_y,
                    model.keep_prob: 0.4,
                })

            # for every epoch test against test data
            acc, loss = sess.run([model.acc, model.loss], feed_dict={
                model.inputs: test_images,
                model.labels: test_labels,
                model.keep_prob: 1.0,
            })
            print('[Epoch-{:d}]: loss: {:.4f}, accuracy: {:.4f}'.format(e, loss, acc))
    return


# ======================================================================================================================
# train with pre-made *.tfrecord
# ======================================================================================================================
def train_with_tfrecord():
    # hyper parameters
    batch_size = 100
    epochs = 20

    # load mnist data
    mnist_tfrecord_dir = './data/mnist-tfrecord'
    training_fn_list = ['mnist-train-00.tfrecord', 'mnist-train-01.tfrecord']
    validate_fn_list = ['mnist-val-00.tfrecord', 'mnist-val-01.tfrecord']
    training_fn_list = [os.path.join(mnist_tfrecord_dir, fn) for fn in training_fn_list]
    validate_fn_list = [os.path.join(mnist_tfrecord_dir, fn) for fn in validate_fn_list]

    filenames_tensor = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filenames_tensor)
    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # create model
    model = CNNModel()

    # start training
    with tf.Session() as sess:
        # run initializer ops
        sess.run(tf.global_variables_initializer())

        # for each epoch
        for e in range(1, epochs + 1):
            sess.run(iterator.initializer, feed_dict={filenames_tensor: training_fn_list})

            while True:
                try:
                    # get data
                    train_x, train_y = sess.run(next_element)

                    # run train operation
                    _ = sess.run(model.train_opt, feed_dict={
                        model.inputs: train_x,
                        model.labels: train_y,
                        model.keep_prob: 0.4,
                    })
                except tf.errors.OutOfRangeError:
                    break

            # for every epoch test against test data
            sess.run(iterator.initializer, feed_dict={filenames_tensor: validate_fn_list})
            accuracies = []
            losses = []
            while True:
                try:
                    # get data
                    test_x, test_y = sess.run(next_element)

                    acc, loss = sess.run([model.acc, model.loss], feed_dict={
                        model.inputs: test_x,
                        model.labels: test_y,
                        model.keep_prob: 1.0,
                    })
                    accuracies.append(acc)
                    losses.append(loss)
                except tf.errors.OutOfRangeError:
                    acc = np.mean(accuracies)
                    loss = np.mean(losses)
                    print('[Epoch-{:d}]: loss: {:.4f}, accuracy: {:.4f}'.format(e, loss, acc))
                    break
    return


def main():
    # select one
    train()

    train_with_tfrecord()
    return


if __name__ == '__main__':
    main()
