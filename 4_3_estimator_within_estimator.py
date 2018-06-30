import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor

from helper import data_input_fn


tf.logging.set_verbosity(tf.logging.INFO)

# ========================================================================================
# https://github.com/tensorflow/tensorflow/issues/14713
# https://stackoverflow.com/questions/45900653/tensorflow-how-to-predict-from-a-savedmodel
# ========================================================================================


# load pretrained model
pretrained_model_dir = './models/high_api/1530363053'
# another_graph = tf.Graph()
# predict_fn = predictor.from_saved_model(export_dir=pretrained_model_dir, graph=another_graph)
predict_fn = predictor.from_saved_model(export_dir=pretrained_model_dir)


def pretrained_predictor(x):
    predict_out = predict_fn({'x': x})
    return predict_out['dense4']


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

    predict_out = tf.py_func(pretrained_predictor, [inputs], tf.float32)

    dense4 = tf.identity(predict_out, name='dense4')
    dense4.set_shape([None, 1024])

    # Logits layer
    # [batch_size, 1024] => [batch_size, 10]
    logits = tf.layers.dense(dense4, units=n_output_classes)

    # ================================
    # prediction & serving mode
    # mode == tf.estimator.ModeKeys.PREDICT == 'infer'
    # ================================
    predicted_classes = tf.argmax(logits, axis=1)
    predictions = {
        'class_id': tf.cast(predicted_classes, dtype=tf.int32),
        'probabilities': tf.nn.softmax(logits),
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

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


def train(model_dir):
    # load mnist data
    mnist_tfrecord_dir = './data/mnist-tfrecord'
    training_fn_list = ['mnist-train-00.tfrecord', 'mnist-train-01.tfrecord']
    validate_fn_list = ['mnist-val-00.tfrecord', 'mnist-val-01.tfrecord']
    training_fn_list = [os.path.join(mnist_tfrecord_dir, fn) for fn in training_fn_list]
    validate_fn_list = [os.path.join(mnist_tfrecord_dir, fn) for fn in validate_fn_list]

    # hyper parameters
    batch_size = 100
    epochs = 20

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
        input_fn=lambda: data_input_fn(validate_fn_list, False, 1, 1))
    print(eval_results)

    return


def test_pretrained(model_servable_dir):
    from tensorflow.examples.tutorials.mnist import input_data

    # =============================================================================
    # here pretrained model is from '4_high_level_api.py' -> create_serving_model()
    # the pretrained model will have input key as 'x' and output key as 'output'
    # =============================================================================

    # load pretrained model
    pretrained_predict_fn = predictor.from_saved_model(export_dir=model_servable_dir)

    # load mnist data
    mnist = input_data.read_data_sets('./data/mnist')
    test_images = np.reshape(mnist.test.images, newshape=[-1, 28, 28, 1])
    test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # lets predict just 10 images
    predictions = pretrained_predict_fn(
        {
            'x': test_images[:10, :, :, :]
        }
    )
    print(predictions['output'])    # outputs: [7 2 1 0 4 1 4 9 5 9]
    print(test_labels[:10])         # outputs: [7 2 1 0 4 1 4 9 5 9]
    return


def inspect_model_dir(model_checkpoint_dir):
    from tensorflow.python.tools import inspect_checkpoint as chkp

    ckpt_prefix = tf.train.latest_checkpoint(model_checkpoint_dir)

    # print all tensors in checkpoint file
    chkp.print_tensors_in_checkpoint_file(ckpt_prefix, tensor_name='', all_tensors=True)
    return


def main():
    # # 0. inspect pretrained model dir
    # model_checkpoint_dir = './models/high_api'
    # inspect_model_dir(model_checkpoint_dir)
    #
    # # 1. test pretrained model
    # model_servable_dir = './models/high_api/1530363053'
    # test_pretrained(model_servable_dir)

    # 2. train the new model which takes pretrained model
    #    since this is toy example, pretrained model is mnist classifier and new model is same mnist classifier
    model_dir = './models/transfered'
    train(model_dir)
    return


if __name__ == '__main__':
    main()
