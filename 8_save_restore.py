import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def autoencoder_network(inputs):
    encoder1 = tf.layers.dense(inputs, units=256, activation=tf.nn.sigmoid, name='ae_enc1')
    encoder2 = tf.layers.dense(encoder1, units=128, activation=tf.nn.sigmoid, name='ae_enc2')
    decoder1 = tf.layers.dense(encoder2, units=256, activation=tf.nn.sigmoid, name='ae_dec1')
    decoder2 = tf.layers.dense(decoder1, units=784, activation=tf.nn.sigmoid, name='ae_dec2')
    return decoder2


def autoencoder_mnist():
    learning_rate = 0.01
    inputs = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='ae_inputs')
    reshaped = tf.layers.flatten(inputs, name='ae_inputs_flatten')

    decoder2 = autoencoder_network(reshaped)
    decoder2 = tf.identity(decoder2, name='ae_output')

    loss = tf.losses.mean_squared_error(reshaped, decoder2)
    train_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)
    return inputs, loss, train_op


def classifier_network(inputs, is_training):
    conv_1 = tf.layers.conv2d(inputs, 32, 5, 2, 'same', name='cl_conv1')
    conv_2 = tf.layers.conv2d(conv_1, 64, 5, 2, 'same', name='cl_conv2')
    flatten = tf.layers.flatten(conv_2, name='cl_flatten')
    dense1 = tf.layers.dense(flatten, 1024, tf.nn.relu, name='cl_dense1')
    dropout1 = tf.layers.dropout(dense1, rate=0.4, training=is_training, name='cl_dropout1')
    logits = tf.layers.dense(dropout1, 10, None, name='cl_logits')
    return logits


def classifier_mnist():
    learning_rate = 0.001
    inputs = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='cl_inputs')
    labels = tf.placeholder(tf.int32, shape=[None], name='cl_labels')
    is_training = tf.placeholder(tf.bool, name='cl_is_training')

    logits = classifier_network(inputs, is_training)
    logits = tf.identity(logits, name='cl_output')

    # loss & optimizer
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    # accuracy computation
    pred = tf.cast(tf.argmax(logits, axis=1), dtype=tf.int32)
    correct_prediction = tf.cast(tf.equal(labels, pred), dtype=tf.float32)
    acc = tf.reduce_mean(correct_prediction, name='cl_accuracy')
    return inputs, labels, is_training, loss, train_op, acc


def train_ae(ae_ckpt_fn):
    tf.reset_default_graph()

    # hyper parameters
    batch_size = 100
    epochs = 20

    # load mnist data
    mnist = input_data.read_data_sets('./data/mnist')
    test_images = np.reshape(mnist.test.images, newshape=[-1, 28, 28, 1])

    # load model
    inputs, loss, train_op = autoencoder_mnist()

    # prepare saver
    saver = tf.train.Saver()

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
                _ = sess.run(train_op, feed_dict={inputs: train_x})

            # for every epoch test against test data
            loss_out = sess.run(loss, feed_dict={inputs: test_images})
            print('[Epoch-{:d}]: test_loss: {:.4f}'.format(e, loss_out))
        save_path = saver.save(sess, ae_ckpt_fn)
        print('Model saved in path: {:s}'.format(save_path))
    return


def train_classifier(ckpt_fn):
    tf.reset_default_graph()

    # hyper parameters
    batch_size = 100
    epochs = 20

    # load mnist data
    mnist = input_data.read_data_sets('./data/mnist')
    test_images = np.reshape(mnist.test.images, newshape=[-1, 28, 28, 1])
    test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # create model
    inputs, labels, is_training, loss, train_op, acc = classifier_mnist()

    # prepare saver
    saver = tf.train.Saver()

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
                _ = sess.run(train_op, feed_dict={
                    inputs: train_x,
                    labels: train_y,
                    is_training: True,
                })

            # for every epoch test against test data
            acc_out, loss_out = sess.run([acc, loss], feed_dict={
                inputs: test_images,
                labels: test_labels,
                is_training: False,
            })
            print('[Epoch-{:d}]: loss: {:.4f}, accuracy: {:.4f}'.format(e, loss_out, acc_out))
        save_path = saver.save(sess, ckpt_fn)
        print('Model saved in path: {:s}'.format(save_path))
    return


def merge_two_graphs(ae_ckpt_fn, cl_ckpt_fn):
    # ==================================================================================================================
    # load mnist test data
    # ==================================================================================================================
    mnist = input_data.read_data_sets('./data/mnist')
    test_images = np.reshape(mnist.test.images, newshape=[-1, 28, 28, 1])
    test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # ==================================================================================================================
    # build merged graph
    # ==================================================================================================================
    tf.reset_default_graph()

    # ae
    inputs = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='inputs')
    reshaped = tf.layers.flatten(inputs, name='inputs_flatten')
    decoder2 = autoencoder_network(reshaped)

    # bridge
    bridge_between_graph = tf.reshape(decoder2, shape=[-1, 28, 28, 1], name='reshape_bridge')

    # cl
    is_training = tf.placeholder(tf.bool, name='is_training')
    logits = classifier_network(bridge_between_graph, is_training)

    # accuracy computation
    labels = tf.placeholder(tf.int32, shape=[None], name='labels')
    pred = tf.cast(tf.argmax(logits, axis=1), dtype=tf.int32)
    correct_prediction = tf.cast(tf.equal(labels, pred), dtype=tf.float32)
    acc = tf.reduce_mean(correct_prediction, name='accuracy')

    # ==================================================================================================================
    # restore each parts
    # ==================================================================================================================
    t_vars1 = [v for v in tf.trainable_variables() if 'ae_' in v.name]
    t_vars2 = [v for v in tf.trainable_variables() if 'cl_' in v.name]
    s1 = tf.train.Saver(t_vars1)
    s2 = tf.train.Saver(t_vars2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        s1.restore(sess, ae_ckpt_fn)
        s2.restore(sess, cl_ckpt_fn)

        acc_out = sess.run(acc, feed_dict={
            inputs: test_images,
            labels: test_labels,
            is_training: False,
        })
        print('Test accuracy: {:.4f}'.format(acc_out))

    return


def main():
    ae_ckpt_fn = '/tmp/auto_encoder.ckpt'
    cl_ckpt_fn = '/tmp/classifier.ckpt'

    # [train phase]
    train_ae(ae_ckpt_fn)
    train_classifier(cl_ckpt_fn)

    # merge
    # some help: https://gist.github.com/marta-sd/ba47a9626ae2dbcc47094c196669fd59
    merge_two_graphs(ae_ckpt_fn, cl_ckpt_fn)
    return


if __name__ == '__main__':
    main()
