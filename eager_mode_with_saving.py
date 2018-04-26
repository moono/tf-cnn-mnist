import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import eager as tfe
from tqdm import trange

tf.logging.set_verbosity(tf.logging.INFO)


class CNNMNIST(tfe.Network):
    def __init__(self):
        super(CNNMNIST, self).__init__(name='cnn-mnist-model')

        # Convolutional Layer #1
        # [batch_size, 28, 28, 1] => [batch_size, 14, 14, 32]
        self.conv1 = self.track_layer(
            tf.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation=tf.nn.relu))
        self.pool1 = self.track_layer(tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2))

        # Convolutional Layer #2
        # [batch_size, 14, 14, 32] => [batch_size, 7, 7, 64]
        self.conv2 = self.track_layer(
            tf.layers.Conv2D(filters=64, kernel_size=5, padding='same', activation=tf.nn.relu))
        self.pool2 = self.track_layer(tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2))

        # Flatten tensor into a batch of vectors
        # [batch_size, 7, 7, 64] => [batch_size, 7 * 7 * 64]
        self.flat3 = self.track_layer(tf.layers.Flatten())

        # Dense Layer with dropout
        # [batch_size, 7 * 7 * 64] => [batch_size, 1024]
        self.dense4 = self.track_layer(tf.layers.Dense(units=1024, activation=tf.nn.relu))
        self.dropout4 = self.track_layer(tf.layers.Dropout(rate=0.4))

        # Logits layer
        # [batch_size, 1024] => [batch_size, 10]
        self.logits = self.track_layer(tf.layers.Dense(units=10))

    def call(self, inputs, is_trainig):
        x = self.pool1(self.conv1(inputs))
        x = self.pool2(self.conv2(x))
        x = self.flat3(x)
        x = self.dropout4(self.dense4(x), training=is_trainig)
        x = self.logits(x)
        return x


def loss_fn(model, is_training, images, labels):
    logits = model(images, is_training)
    onehot_labels = tf.one_hot(labels, depth=10)
    return tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)


def train():
    # Enable eager execution (had to set device_policy=tfe.DEVICE_PLACEMENT_SILENT, I don't know why)
    tfe.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)

    # check gpu availability
    device = '/gpu:0'
    if tfe.num_gpus() <= 0:
        device = '/cpu:0'

    # where to save the trained model
    checkpoint_directory = './eager_models'
    checkpoint_prefix = os.path.join(checkpoint_directory, 'eager-mnist')

    # hyper parameters
    batch_size = 100
    epochs = 20
    learning_rate = 0.001
    is_training = True

    # load mnist data
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    val_images = mnist.test.images
    val_labels = mnist.test.labels
    val_images = tf.reshape(val_images, shape=[-1, 28, 28, 1])
    val_labels = tf.cast(val_labels, dtype=tf.int32)

    # create model
    cnn_mnist_model = CNNMNIST()

    # prepare optimizer
    val_grad = tfe.implicit_value_and_gradients(loss_fn)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    global_step = tf.train.get_or_create_global_step()

    # prepare saver - need to run dummy data first...
    _ = cnn_mnist_model(tf.zeros([1, 28, 28, 1], dtype=tf.float32), False)
    saver = tfe.Saver(var_list=cnn_mnist_model.variables)

    # for loss savings
    loss_at_steps = []

    # for accuracy
    last_acc = 0.0

    # wrap with available device
    with tf.device(device):
        for e in range(epochs):
            t = trange(mnist.train.num_examples // batch_size)
            for ii in t:
                t.set_description('{:04d}/{:04d}: '.format(e + 1, epochs))

                # get train data
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                batch_x = tf.reshape(batch_x, shape=[-1, 28, 28, 1])

                # get loss related values & (gradients & vars)
                loss_val, grad_vars = val_grad(cnn_mnist_model, is_training, batch_x, batch_y)

                # apply gradient via pre-defined optimizer
                optimizer.apply_gradients(grad_vars, global_step=global_step)

                # save loss
                loss_at_steps.append(np.asscalar(loss_val.numpy()))

                # display current losses
                if ii % 5 == 0:
                    t.set_postfix(loss=loss_val.numpy(), accuracy=last_acc)

            # validation results at every epoch
            logits = cnn_mnist_model(val_images, False)

            pred = tf.cast(tf.argmax(logits, axis=1), dtype=tf.int32)
            correct_prediction = tf.cast(tf.equal(val_labels, pred), dtype=tf.float32)
            acc = tf.reduce_mean(correct_prediction)
            last_acc = acc.numpy()

            if e % 10 == 0:
                saver.save(checkpoint_prefix, global_step=global_step)

        saver.save(checkpoint_prefix, global_step=global_step)
    return


def evaluate():
    # Enable eager execution (had to set device_policy=tfe.DEVICE_PLACEMENT_SILENT, I don't know why)
    tfe.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)

    # check gpu availability
    device = '/gpu:0'
    if tfe.num_gpus() <= 0:
        device = '/cpu:0'

    # where to save the trained model
    checkpoint_directory = './eager_models'

    # load mnist data
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    val_images = mnist.test.images
    val_labels = mnist.test.labels
    val_images = tf.reshape(val_images, shape=[-1, 28, 28, 1])
    val_labels = tf.cast(val_labels, dtype=tf.int32)

    # create model
    cnn_mnist_model = CNNMNIST()

    # restore model - run dummy data first
    _ = cnn_mnist_model(tf.zeros([1, 28, 28, 1], dtype=tf.float32), False)
    saver = tfe.Saver(var_list=cnn_mnist_model.variables)
    saver.restore(tf.train.latest_checkpoint(checkpoint_directory))

    # wrap with available device
    with tf.device(device):
        # evaluate
        logits = cnn_mnist_model(val_images, False)

        pred = tf.cast(tf.argmax(logits, axis=1), dtype=tf.int32)
        correct_prediction = tf.cast(tf.equal(val_labels, pred), dtype=tf.float32)
        acc = tf.reduce_mean(correct_prediction)

        print('accuracy: {:.4f}'.format(acc.numpy()))
    return


# under construction
# see: https://stackoverflow.com/questions/47852516/tensorflow-eager-mode-how-to-restore-a-model-from-a-checkpoint
def evaluate_graph_mode():
    # create model
    cnn_mnist_model = CNNMNIST()
    _ = cnn_mnist_model(tf.zeros([1, 28, 28, 1], dtype=tf.float32), False)

    # where to save the trained model
    checkpoint_directory = './eager_models'

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_directory))
        print('')

    return


def main():
    # train()
    # evaluate()
    evaluate_graph_mode()
    return


if __name__ == '__main__':
    main()
