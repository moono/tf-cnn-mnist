import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from helper import data_input_fn

tf.logging.set_verbosity(tf.logging.INFO)


# model_fn with tf.estimator.Estimator function signature
def cnn_model_fn(features, labels, mode, params):
    """
    :param features: dictionary with single key 'x' which represents input images
    :param labels: ground truth label
    :param mode: tensorflow mode - TRAIN, PREDICT, EVAL
    :param params: dictionay of additional parameter
    :return: tf.estimator.EstimatorSpec
    """

    # ================================
    # common operations for all modes
    # ================================
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    input_size = 28
    n_output_classes = 10
    if params is not None:
        input_size = params['input_size']
        n_output_classes = params['n_output_classes']
    inputs = tf.reshape(features['x'], shape=[-1, input_size, input_size, 1])

    # Convolutional Layer #1
    # [batch_size, 28, 28, 1] => [batch_size, 14, 14, 32]
    conv1 = tf.layers.conv2d(inputs, filters=32, kernel_size=5, padding='same', activation=tf.nn.relu)
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
    logits = tf.layers.dense(dropout4, units=n_output_classes)

    # ================================
    # prediction & serving mode
    # mode == tf.estimator.ModeKeys.PREDICT == 'infer'
    # ================================
    predicted_classes = tf.argmax(logits, axis=1)
    predictions = {
        'class_id': tf.cast(predicted_classes, dtype=tf.int32),
        'probabilities': tf.nn.softmax(logits),
    }
    # export output must be one of tf.estimator.export. ... class NOT a Tensor
    export_outputs = {
        'output_classes': tf.estimator.export.PredictOutput(predictions['class_id']),
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # compute loss
    # labels: integer 0 ~ 9
    # logits: score not probability
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # compute evaluation metric
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')
    metrics = {'accuracy': accuracy}            # during evaluation
    tf.summary.scalar('accuracy', accuracy[1])  # during training

    # ================================
    # evaluation mode
    # ================================
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)

    # ================================
    # training mode
    # ================================
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_ops = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_ops)


def train(fresh_training, model_dir):
    # load mnist data
    mnist_tfrecord_dir = './data/mnist-tfrecord'
    training_fn_list = ['mnist-train-00.tfrecord', 'mnist-train-01.tfrecord']
    validate_fn_list = ['mnist-val-00.tfrecord', 'mnist-val-01.tfrecord']
    training_fn_list = [os.path.join(mnist_tfrecord_dir, fn) for fn in training_fn_list]
    validate_fn_list = [os.path.join(mnist_tfrecord_dir, fn) for fn in validate_fn_list]

    # hyper parameters
    batch_size = 100
    epochs = 20

    # clear saved model directory
    if fresh_training:
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir)

    # create run config for estimator
    run_config = tf.estimator.RunConfig(keep_checkpoint_max=3)

    # create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=model_dir,
        config=run_config,
        params={
            'input_size': 28,
            'n_output_classes': 10,
        },
        warm_start_from=None
    )

    # tf.estimator.train() && tf.estimator.eval()
    # Will save checkpoint at every 600 (defaults) seconds unless specified in RunConfig.
    # If you want to save checkpoint at the end of every epoch, use save_checkpoints_steps in RunConfig.
    # Use epochs in input_fn to control training stop condition.
    # Will automatically add summary for loss, eval_metric, global step/sec.

    # train model
    mnist_classifier.train(
        input_fn=lambda: data_input_fn(training_fn_list, True, batch_size, epochs),
        hooks=None,
        steps=None,
        max_steps=None
    )

    # evaluate the model and print results
    # hooks not working for evaluation?
    eval_results = mnist_classifier.evaluate(
        input_fn=lambda: data_input_fn(validate_fn_list, False, 1, epochs=1))
    print(eval_results)

    return


def inference(model_dir):
    # load mnist data
    mnist = input_data.read_data_sets('./data/mnist')
    test_images = np.reshape(mnist.test.images, newshape=[-1, 28, 28, 1])
    test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Load trained Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=model_dir,
        config=None,
        params={
            'input_size': 28,
            'n_output_classes': 10,
        },
        warm_start_from=None
    )

    # input function with numpy
    batch_size = 100
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': test_images},
        y=test_labels,
        batch_size=batch_size,
        num_epochs=1,
        shuffle=False,
    )

    # estimator's predict returns generator
    predicted_results = mnist_classifier.predict(input_fn=predict_input_fn)

    # need to iterate it and extract & print result
    for ii, results in enumerate(predicted_results):
        predicted_label = results['class_id']
        print(predicted_label)

        # just print 10 predictions
        if ii > 10:
            break

    return


# function used to map input for tensorflow serving system
def serving_input_receiver_fn():
    # dictionary used to input when serving
    # here 'x' is the serving input name
    inputs = {
        'x': tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='serving_input_image_x')
    }
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def create_serving_model(model_dir):
    # Load trained Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=model_dir,
        config=None,
        params={
            'input_size': 28,
            'n_output_classes': 10,
        },
        warm_start_from=None
    )

    # below function will save servable files to 'model_dir'
    # which will be named with current time stamp
    # you can inspect saved servable model with saved_model_cli
    # ex) saved_model_cli show --dir model_dir --all
    # default model name: 'serve'
    # default signature name: 'serving_default' or the name you specified in model_fn's export_outputs dict
    # default output name: 'output'
    mnist_classifier.export_savedmodel(model_dir, serving_input_receiver_fn=serving_input_receiver_fn)
    return


def main():
    model_dir = './models/high_api'

    # 1. train the model
    train(fresh_training=True, model_dir=model_dir)

    # 2. test prediction with current saved model files
    inference(model_dir)

    # 3. make tensorflow serving files
    create_serving_model(model_dir)
    return


if __name__ == '__main__':
    main()
