import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import trange


class CNNModel(object):
    def __init__(self):
        # inputs
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
        self.inputs = tf.placeholder(tf.float32, shape=[None, 784], name='mnist_input')
        self.labels = tf.placeholder(tf.float32, shape=[None], name='mnist_label')

        # preprocess label
        self.int32_labels = tf.cast(self.labels, tf.int32)
        self.onehot_labels = tf.one_hot(indices=self.int32_labels, depth=10)

        # define layers
        # Reshape inputs to 4-D tensor: [batch_size, width, height, channels]
        self.reshape0 = tf.reshape(self.inputs, shape=[-1, 28, 28, 1])

        # Convolutional Layer #1
        # [batch_size, 28, 28, 1] => [batch_size, 14, 14, 32]
        self.conv1 = tf.layers.conv2d(self.reshape0, filters=32, kernel_size=5, padding='same', activation=tf.nn.relu)
        self.pool1 = tf.layers.max_pooling2d(self.conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2
        # [batch_size, 14, 14, 32] => [batch_size, 7, 7, 64]
        self.conv2 = tf.layers.conv2d(self.pool1, filters=64, kernel_size=5, padding='same', activation=tf.nn.relu)
        self.pool2 = tf.layers.max_pooling2d(self.conv2, pool_size=[2, 2], strides=2)

        # Flatten tensor into a batch of vectors
        # [batch_size, 7, 7, 64] => [batch_size, 7 * 7 * 64]
        self.flat3 = tf.layers.flatten(self.pool2)

        # Dense Layer with dropout
        # [batch_size, 7 * 7 * 64] => [batch_size, 1024]
        self.dense4 = tf.layers.dense(self.flat3, units=1024, activation=tf.nn.relu)
        self.dropout4 = tf.layers.dropout(self.dense4, rate=0.4, training=self.is_training)

        # Logits layer
        # [batch_size, 1024] => [batch_size, 10]
        self.logits = tf.layers.dense(self.dropout4, units=10)

        # loss
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.onehot_labels, logits=self.logits)

        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        self.train_opt = optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())

        # model outputs
        self.pred = tf.cast(tf.argmax(self.logits, axis=1), dtype=tf.int32)
        self.correct_prediction = tf.cast(tf.equal(self.int32_labels, self.pred), dtype=tf.float32)
        self.acc = tf.reduce_mean(self.correct_prediction)
        self.probs = tf.nn.softmax(self.logits)
        return


def train():
    # hyper parameters
    batch_size = 100
    epochs = 20

    # load mnist data
    mnist = input_data.read_data_sets('mnist-data', one_hot=False)
    test_images = mnist.test.images
    test_labels = mnist.test.labels

    # create model
    model = CNNModel()

    with tf.Session() as sess:
        # run initializer ops
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            t = trange(mnist.train.num_examples // batch_size)
            for ii in t:
                t.set_description('{:04d}/{:04d}: '.format(e + 1, epochs))

                # get data
                batch_x, batch_y = mnist.train.next_batch(batch_size)

                _ = sess.run(model.train_opt, feed_dict={
                    model.inputs: batch_x,
                    model.labels: batch_y,
                    model.is_training: True
                })

                if ii % 100 == 0:
                    acc, loss = sess.run([model.acc, model.loss], feed_dict={
                        model.inputs: test_images,
                        model.labels: test_labels,
                        model.is_training: False
                    })
                    t.set_postfix(loss=loss, accuracy=acc)
    return


def main():
    train()
    return


if __name__ == '__main__':
    main()
