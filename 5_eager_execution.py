import os
import shutil
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.examples.tutorials.mnist import input_data

from helper import parse_tfrecord


# ======================================================================================================================
# Model definition
# ======================================================================================================================
# define model via tf.keras.Model
# tfe.Network is deprecated in tensorflow r1.9
class CNNMNIST(tf.keras.Model):
    def __init__(self):
        super(CNNMNIST, self).__init__()

        # Convolutional Layer #1
        # [batch_size, 28, 28, 1] => [batch_size, 14, 14, 32]
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation=tf.nn.relu)
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)

        # Convolutional Layer #2
        # [batch_size, 14, 14, 32] => [batch_size, 7, 7, 64]
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, padding='same', activation=tf.nn.relu)
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)

        # Flatten tensor into a batch of vectors
        # [batch_size, 7, 7, 64] => [batch_size, 7 * 7 * 64]
        self.flat3 = tf.keras.layers.Flatten()

        # Dense Layer with dropout
        # [batch_size, 7 * 7 * 64] => [batch_size, 1024]
        self.dense4 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dropout4 = tf.keras.layers.Dropout(rate=0.4)

        # Logits layer
        # [batch_size, 1024] => [batch_size, 10]
        self.logits = tf.keras.layers.Dense(units=10)

    # when calling the model must use keyword argument 'training='
    # ex) model = CNNMNIST()
    #     output_logit = model(images, training=False)
    def call(self, inputs, training=False):
        x = self.pool1(self.conv1(inputs))
        x = self.pool2(self.conv2(x))
        x = self.flat3(x)
        x = self.dropout4(self.dense4(x), training=training)
        x = self.logits(x)
        return x


# ======================================================================================================================
# Training
# ======================================================================================================================
def loss_fn(model, is_training, images, labels):
    logits = model(images, training=is_training)
    onehot_labels = tf.one_hot(labels, depth=10)
    return tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)


def grad_fn(model, is_training, images, labels):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, is_training, images, labels)
    return tape.gradient(loss, model.variables)


def train(model_dir):
    # must enable eager execution first
    tf.enable_eager_execution()

    # hyper parameters
    batch_size = 100
    epochs = 20
    learning_rate = 0.001
    is_training = True

    # load mnist data
    mnist = input_data.read_data_sets('./data/mnist')
    test_images = np.reshape(mnist.test.images, newshape=[-1, 28, 28, 1])
    test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # where to save the trained model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    checkpoint_prefix = os.path.join(model_dir, 'model')

    # create model
    model = CNNMNIST()

    # prepare optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    # prepare saver - need to run dummy data first...
    _ = model(tf.zeros([1, 28, 28, 1], dtype=tf.float32), training=False)
    saver = tfe.Saver(var_list=model.variables)

    # start training
    for e in range(1, epochs + 1):
        for ii in range(mnist.train.num_examples // batch_size):
            # get train data
            train_x, train_y = mnist.train.next_batch(batch_size)
            train_x = tf.reshape(train_x, shape=[-1, 28, 28, 1])

            # optimize model
            grads = grad_fn(model, is_training, train_x, train_y)
            optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())

        # track progress
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()
        epoch_loss_avg(loss_fn(model, False, test_images, test_labels))  # add current batch loss

        # compare predicted label to actual label
        epoch_accuracy(tf.argmax(model(test_images, training=False), axis=1, output_type=tf.int32), test_labels)

        # save loss & accuracy
        # for every epoch test against test data
        loss = epoch_loss_avg.result()
        acc = epoch_accuracy.result()
        print('[Epoch-{:d}]: loss: {:.4f}, accuracy: {:.4f}'.format(e, loss, acc))

        if e % 5 == 0:
            saver.save(checkpoint_prefix, global_step=tf.train.get_or_create_global_step())
    saver.save(checkpoint_prefix, global_step=tf.train.get_or_create_global_step())
    return


def get_dataset(batch_size, tfrecord_list):
    dataset = tf.data.TFRecordDataset(tfrecord_list)
    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.batch(batch_size)
    return dataset


def train_with_tfrecord(model_dir):
    # must enable eager execution first
    tf.enable_eager_execution()

    # hyper parameters
    batch_size = 100
    epochs = 20
    learning_rate = 0.001
    is_training = True

    # load mnist data
    mnist_tfrecord_dir = './data/mnist-tfrecord'
    training_fn_list = ['mnist-train-00.tfrecord', 'mnist-train-01.tfrecord']
    validate_fn_list = ['mnist-val-00.tfrecord', 'mnist-val-01.tfrecord']
    training_fn_list = [os.path.join(mnist_tfrecord_dir, fn) for fn in training_fn_list]
    validate_fn_list = [os.path.join(mnist_tfrecord_dir, fn) for fn in validate_fn_list]

    train_dataset = get_dataset(batch_size, training_fn_list)
    test_dataset = get_dataset(batch_size, validate_fn_list)

    # where to save the trained model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    checkpoint_prefix = os.path.join(model_dir, 'model')

    # create model
    model = CNNMNIST()

    # prepare optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    # prepare saver - need to run dummy data first...
    _ = model(tf.zeros([1, 28, 28, 1], dtype=tf.float32), training=False)
    saver = tfe.Saver(var_list=model.variables)

    # start training
    for e in range(1, epochs + 1):
        # get train data
        for train_x, train_y in train_dataset:
            # optimize model
            grads = grad_fn(model, is_training, train_x, train_y)
            optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())

        # get test data
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()
        for test_images, test_labels in test_dataset:
            # Track progress
            epoch_loss_avg(loss_fn(model, False, test_images, test_labels))  # add current batch loss

            # compare predicted label to actual label
            epoch_accuracy(tf.argmax(model(test_images, training=False), axis=1, output_type=tf.int32), test_labels)

        # save loss & accuracy
        # for every epoch test against test data
        loss = epoch_loss_avg.result()
        acc = epoch_accuracy.result()
        print('[Epoch-{:d}]: loss: {:.4f}, accuracy: {:.4f}'.format(e, loss, acc))

        if e % 5 == 0:
            saver.save(checkpoint_prefix, global_step=tf.train.get_or_create_global_step())
    saver.save(checkpoint_prefix, global_step=tf.train.get_or_create_global_step())
    return


# ======================================================================================================================
# Evaluations
# ======================================================================================================================
def evaluate(model_dir):
    # must enable eager execution first
    tf.enable_eager_execution()

    # load mnist data
    mnist = input_data.read_data_sets('./data/mnist')
    test_images = np.reshape(mnist.test.images, newshape=[-1, 28, 28, 1])
    test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # create model
    model = CNNMNIST()

    # restore model - run dummy data first
    _ = model(tf.zeros([1, 28, 28, 1], dtype=tf.float32), training=False)
    saver = tfe.Saver(var_list=model.variables)
    saver.restore(tf.train.latest_checkpoint(model_dir))

    # evaluate
    logits = model(test_images, training=False)
    pred = tf.cast(tf.argmax(logits, axis=1), dtype=tf.int32)
    correct_prediction = tf.cast(tf.equal(test_labels, pred), dtype=tf.float32)
    acc = tf.reduce_mean(correct_prediction)
    print('test accuracy: {:.4f}'.format(acc.numpy()))
    return


# see: https://stackoverflow.com/questions/47852516/tensorflow-eager-mode-how-to-restore-a-model-from-a-checkpoint
def evaluate_graph_mode(model_dir):
    # load mnist data
    mnist = input_data.read_data_sets('./data/mnist')
    test_images = np.reshape(mnist.test.images, newshape=[-1, 28, 28, 1])
    test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # create model
    model = CNNMNIST()

    # add mode graph nodes
    inputs = tf.placeholder(tf.float32, [None, 28, 28, 1], name='new_inputs')
    logits = model(inputs, training=False)
    pred = tf.cast(tf.argmax(logits, axis=1), dtype=tf.int32, name='prediction')
    correct_prediction = tf.cast(tf.equal(test_labels, pred), dtype=tf.float32)
    acc = tf.reduce_mean(correct_prediction)

    # load trained model to start
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))

        print('Inputs name: {:s}'.format(inputs.name))  # new_inputs:0
        print('Output name: {:s}'.format(logits.name))  # cnnmnist/dense_1/BiasAdd:0
        print('Output name: {:s}'.format(pred.name))    # prediction:0

        result = sess.run(acc, feed_dict={inputs: test_images})
        print(result)
    return


# ======================================================================================================================
# Convert for tensorflow serving from eagerly save model
# ======================================================================================================================
def convert_for_serving(model_dir):
    # start building graph
    tf.reset_default_graph()

    # prepare export path
    model_version = 1
    export_path = os.path.join(model_dir, str(model_version))

    # SavedModelBuilder will create the directory if it does not exist
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    # try to restore & reorganize network
    model = CNNMNIST()
    inputs = tf.placeholder(tf.float32, [None, 28, 28, 1], name='inputs')
    logits = model(inputs, training=False)
    pred = tf.cast(tf.argmax(logits, axis=1), dtype=tf.int32, name='prediction')

    # retore from...
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # restore network
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))

        inputs_placeholder = tf.get_default_graph().get_tensor_by_name('inputs:0')
        output_tensor = tf.get_default_graph().get_tensor_by_name('prediction:0')

        # build tensor info for exporting
        tensor_info_inputs = tf.saved_model.utils.build_tensor_info(inputs_placeholder)
        tensor_info_output = tf.saved_model.utils.build_tensor_info(output_tensor)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    'inputs': tensor_info_inputs,
                },
                outputs={
                    'output': tensor_info_output
                },
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'prediction': prediction_signature,
            },
            legacy_init_op=legacy_init_op)

        builder.save()
    return


def main():
    # model directory
    model_dir = './models/eager'

    # training flag to avoid error
    is_training = True

    if is_training:
        # clear model directory
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # train (choose one)
        train(model_dir)
        train_with_tfrecord(model_dir)
    else:
        # evaluate (choose one)
        evaluate(model_dir)
        evaluate_graph_mode(model_dir)
        convert_for_serving(model_dir)
    return


if __name__ == '__main__':
    main()
