import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor

tf.logging.set_verbosity(tf.logging.INFO)

# ========================================================================================
# https://github.com/tensorflow/tensorflow/issues/14713
# https://stackoverflow.com/questions/45900653/tensorflow-how-to-predict-from-a-savedmodel
# ========================================================================================


# load pretrained model
pretrained_model_dir = './models/1530189906'
another_graph = tf.Graph()
predict_fn = predictor.from_saved_model(export_dir=pretrained_model_dir, graph=another_graph)


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


def parse_tfrecord(raw_record):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/class/label': tf.FixedLenFeature((), tf.int64),
    }

    # parse feature
    parsed = tf.parse_single_example(raw_record, keys_to_features)

    label = tf.cast(parsed['image/class/label'], tf.int32)

    # parse image
    image = tf.image.decode_png(parsed['image/encoded'])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    return image, label


def data_input_fn(data_fn, n_images, is_training, batch_size, epochs):
    dataset = tf.data.TFRecordDataset(data_fn)

    if is_training:
        dataset = dataset.shuffle(buffer_size=n_images)

    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    features = {
        'x': images,
    }
    return features, labels


def train():

    # assert predict_fn.graph is tf.get_default_graph()

    # load mnist data
    train_dataset_fn_list = ['./data/mnist-train-00.tfrecord', './data/mnist-train-01.tfrecord']
    eval_dataset_fn_list = ['./data/mnist-val-00.tfrecord', './data/mnist-val-01.tfrecord']
    n_train_images = 55000
    n_eval_images = 10000

    # hyper parameters
    new_model_dir = './models_new'
    batch_size = 100
    epochs = 20

    # create run config for estimator
    run_config = tf.estimator.RunConfig(keep_checkpoint_max=3)

    # create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=new_model_dir,
        config=run_config,
        params={
            'input_size': 28,
            'n_output_classes': 10,
            # 'pretrained_model': pretrained_mnist_classifier,
            # 'model_a_predict_fn': predict_fn
            # 'pretrained_model_dir': pretrained_model_dir,
        },
        warm_start_from=None
    )

    # train model
    mnist_classifier.train(
        input_fn=lambda: data_input_fn(train_dataset_fn_list, n_train_images, True, batch_size, epochs),
        hooks=None,
        steps=None,
        max_steps=None
    )

    # evaluate the model and print results
    # hooks not working for evaluation?
    eval_results = mnist_classifier.evaluate(
        input_fn=lambda: data_input_fn(eval_dataset_fn_list, n_eval_images, False, 1, 1))
    print(eval_results)

    return


def test_pretrained():
    # ==========================================================================
    # There will be pretrained model in './models/1524893870'
    # here pretrained model is from 'higher_api.py' -> create_serving_model()
    # the pretrained model will have input key as 'x' and output key as 'output'
    # ==========================================================================

    # load pretrained model
    pretrained_model_dir = './models/1524893870'
    predict_fn = predictor.from_saved_model(export_dir=pretrained_model_dir)

    # load test mnist data
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    val_images = mnist.test.images
    val_labels = mnist.test.labels
    val_images = np.reshape(val_images, newshape=(-1, 28, 28, 1))

    # lets predict just 10 images
    predictions = predict_fn(
        {
            'x': val_images[:10, :, :, :]
        }
    )
    print(predictions['output'])    # outputs: [7 2 1 0 4 1 4 9 5 9]
    print(val_labels[:10])          # outputs: [7 2 1 0 4 1 4 9 5 9]
    return


def inspect_model_dir():
    from tensorflow.python.tools import inspect_checkpoint as chkp

    model_dir = './models'
    ckpt_prefix = tf.train.latest_checkpoint(model_dir)

    # print all tensors in checkpoint file
    chkp.print_tensors_in_checkpoint_file(ckpt_prefix, tensor_name='', all_tensors=True)
    return


def main():
    # # 0. inspect pretrained model dir
    # inspect_model_dir()

    # # 1. test pretrained model
    # test_pretrained()

    # 2. train the new model which takes pretrained model
    #    since this is toy example, pretrained model is mnist classifier and new model is same mnist classifier
    train()
    return


if __name__ == '__main__':
    main()
