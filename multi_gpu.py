import numpy as np
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_gpus', 1, """How many GPUs to use.""")
tf.app.flags.DEFINE_integer('batch_size', 1024, """...""")


def model(inputs, reuse, is_training):
    # define layers
    with tf.variable_scope('cnn_mnist', reuse=reuse):
        # Reshape inputs to 4-D tensor: [batch_size, width, height, channels]
        reshape0 = tf.reshape(inputs, shape=[-1, 28, 28, 1])

        # Convolutional Layer #1
        # [batch_size, 28, 28, 1] => [batch_size, 14, 14, 32]
        conv1 = tf.layers.conv2d(reshape0, filters=32, kernel_size=5, padding='same', activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2
        # [batch_size, 14, 14, 32] => [batch_size, 7, 7, 64]
        conv2 = tf.layers.conv2d(pool1, filters=64, kernel_size=5, padding='same', activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)

        # Flatten tensor into a batch of vectors
        # [batch_size, 7, 7, 64] => [batch_size, 7 * 7 * 64]
        flat3 = tf.layers.flatten(pool2)

        # Dense Layer with dropout
        # [batch_size, 7 * 7 * 64] => [batch_size, 1024]
        dense4 = tf.layers.dense(flat3, units=1024, activation=tf.nn.relu)
        dropout4 = tf.layers.dropout(dense4, rate=0.4, training=is_training)

        # Logits layer
        # [batch_size, 1024] => [batch_size, 10]
        logits = tf.layers.dense(dropout4, units=10)
    return logits


def train(mnist, num_gpus, learning_rate, epochs, batch_size):
    # input placeholders
    # is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    inputs = tf.placeholder(tf.float32, shape=[None, 784], name='mnist_input')
    labels = tf.placeholder(tf.float32, shape=[None, 10], name='mnist_label')

    # split inputs
    inputs_splitted = tf.split(inputs, num_gpus)
    labels_splitted = tf.split(labels, num_gpus)

    # collect losses
    losses = []
    for gpu_id in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
            reuse = gpu_id > 0
            with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                output_logits = model(inputs_splitted[gpu_id], reuse, is_training=True)
                cost = tf.nn.softmax_cross_entropy_with_logits(logits=output_logits, labels=labels_splitted[gpu_id])
                losses.append(cost)

                if gpu_id == 0:
                    logits_testing = model(inputs, reuse=True, is_training=False)
                    correct_pred = tf.equal(tf.argmax(logits_testing, 1), tf.argmax(labels, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # optimizing
    loss = tf.reduce_mean(tf.concat(losses, axis=0))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, colocate_gradients_with_ops=True)

    # start training
    loss_train = []
    acc_test = []
    prev_acc = 0
    start = time.time()
    with tf.Session() as sess:
        # run initializer ops
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            for i in range(mnist.train.num_examples // batch_size):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)

                _, loss_val = sess.run([optimizer, loss], feed_dict={inputs: batch_xs, labels: batch_ys})
                loss_train.append(loss_val)
                acc_test.append(prev_acc)

            acc_val = sess.run(accuracy, feed_dict={inputs: mnist.test.images, labels: mnist.test.labels})
            prev_acc = acc_val
    elapsed = time.time() - start

    # prepare plotter
    title = 'NUM_GPU - {}, BATCH_SIZE - {}'.format(num_gpus, batch_size)
    save_fn = '{:s}.data'.format(title)

    x = np.array([i + 1 for i in range(len(loss_train))])
    loss_train = np.array(loss_train)
    acc_test = np.array(acc_test)
    np.savetxt(save_fn, (x, loss_train, acc_test), fmt='%.4e')

    return elapsed


def main(argv=None):
    # data
    mnist = input_data.read_data_sets('mnist-data', one_hot=True)

    # Training Parameters
    num_gpus = FLAGS.num_gpus
    epochs = 10
    learning_rate = 0.001
    batch_size = FLAGS.batch_size

    elapsed = train(mnist, num_gpus, learning_rate, epochs, batch_size)
    print('Total elapsed: {:.4f}s, n_gpu: {}, bs: {}'.format(elapsed, num_gpus, batch_size))
    return


if __name__ == '__main__':
    tf.app.run()
