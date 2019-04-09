# https://stackoverflow.com/questions/45206910/tensorflow-exponential-moving-average

import os
import shutil
import pprint
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from helper import data_input_fn

tf.logging.set_verbosity(tf.logging.INFO)


def network(inputs, n_output_classes, is_training):
    with tf.variable_scope('cnn', reuse=tf.AUTO_REUSE):
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
    return logits


# model_fn with tf.estimator.Estimator function signature
def cnn_model_fn(features, labels, mode, params):
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

    logits = network(inputs, n_output_classes, is_training)

    # apply exponential moving average
    ema_vars = [v for v in tf.trainable_variables() if v.name.startswith('cnn')]
    ema = tf.train.ExponentialMovingAverage(decay=0.998)
    ema_op = ema.apply(ema_vars)
    # ema_val = ema.average(ema_vars)
    pprint.pprint(ema.variables_to_restore())

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
    with tf.control_dependencies([ema_op]):
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # compute evaluation metric
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')
    metrics = {'accuracy': accuracy}            # during evaluation
    tf.summary.scalar('accuracy', accuracy[1])  # during training

    # ================================
    # evaluation mode
    # ================================
    if mode == tf.estimator.ModeKeys.EVAL:
        model_dir = params['trained_model_dir']
        variables_to_restore = ema.variables_to_restore()
        tf.train.init_from_checkpoint(model_dir, variables_to_restore)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)

    # ================================
    # training mode
    # ================================
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    with tf.control_dependencies([ema_op]):
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
    run_config = tf.estimator.RunConfig(keep_checkpoint_max=1)

    # create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=model_dir,
        config=run_config,
        params={
            'input_size': 28,
            'n_output_classes': 10,
            'trained_model_dir': model_dir,
        },
        warm_start_from=None
    )

    # train model
    mnist_classifier.train(
        input_fn=lambda: data_input_fn(training_fn_list, True, batch_size, epochs),
        hooks=None,
        steps=None,
        max_steps=None
    )

    # evaluate the model and print results
    # hooks not working for evaluation?
    eval_results = mnist_classifier.evaluate(input_fn=lambda: data_input_fn(validate_fn_list, False, 1, epochs=1))
    print(eval_results)

    return


def inference(empty_model_dir, trained_model_dir):
    # load mnist data
    mnist = input_data.read_data_sets('./data/mnist')
    test_images = np.reshape(mnist.test.images, newshape=[-1, 28, 28, 1])
    test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # var_name_to_prev_var_name = {
    #     'cnn/conv2d/bias': 'cnn/conv2d/bias/ExponentialMovingAverage',
    #     'cnn/conv2d/kernel': 'cnn/conv2d/kernel/ExponentialMovingAverage',
    #     'cnn/conv2d_1/bias':'cnn/conv2d_1/bias/ExponentialMovingAverage',
    #     'cnn/conv2d_1/kernel': 'cnn/conv2d_1/kernel/ExponentialMovingAverage',
    #     'cnn/dense/bias': 'cnn/dense/bias/ExponentialMovingAverage',
    #     'cnn/dense/kernel': 'cnn/dense/kernel/ExponentialMovingAverage',
    #     'cnn/dense_1/bias': 'cnn/dense_1/bias/ExponentialMovingAverage',
    #     'cnn/dense_1/kernel': 'cnn/dense_1/kernel/ExponentialMovingAverage',
    # }
    # ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=model_dir2,
    #                                     var_name_to_prev_var_name=var_name_to_prev_var_name)

    # Load trained Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=empty_model_dir,
        config=None,
        params={
            'input_size': 28,
            'n_output_classes': 10,
            'trained_model_dir': trained_model_dir,
        },
        # warm_start_from=ws
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

    # evaluate
    eval_results = mnist_classifier.evaluate(input_fn=predict_input_fn)
    print(eval_results)

    return


def main():
    model_dir = './models/high_api'

    # 1. train the model
    train(fresh_training=True, model_dir=model_dir)

    # 2. test prediction with current saved model files
    inference(empty_model_dir='./models/high_api/empty', trained_model_dir=model_dir)
    return


if __name__ == '__main__':
    main()
