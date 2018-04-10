import os
import shutil
import numpy as np
import tensorflow as tf

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

    # # add items to log
    # input_label_copy = tf.identity(labels[0], name='first_label_item')

    # ================================
    # prediction & serving mode
    # mode == tf.estimator.ModeKeys.PREDICT == 'infer'
    # ================================
    predicted_classes = tf.argmax(logits, axis=1)
    predictions = {
        'class_id': tf.cast(predicted_classes, dtype=tf.int32),
        'probabilities': tf.nn.softmax(logits),
        'logits': logits,
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


def parse_tfrecord(raw_record):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/class/label': tf.FixedLenFeature((), tf.int64),
    }

    # parse feature
    parsed = tf.parse_single_example(raw_record, keys_to_features)

    label = tf.cast(parsed['image/class/label'], tf.int32)

    image = tf.image.decode_png(parsed['image/encoded'])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label


def data_input_fn(data_fn, n_images, is_training, num_epochs, batch_size):
    dataset = tf.data.TFRecordDataset(data_fn)

    if is_training:
        dataset = dataset.shuffle(buffer_size=n_images)

    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    features = {
        'x': images,
    }
    return features, labels


def train(fresh_training, option='simple'):
    if option not in ['simple', 'utility_function']:
        raise ValueError('option must be either simple or utility_function')

    # load mnist data
    train_dataset_fn_list = ['./data/mnist-train-00.tfrecord', './data/mnist-train-01.tfrecord']
    eval_dataset_fn_list = ['./data/mnist-val-00.tfrecord', './data/mnist-val-01.tfrecord']
    n_train_images = 55000
    n_eval_images = 10000

    # hyper parameters
    model_dir = './models'
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

    # note: when using estimator.train & estimator.eval, use epochs in input_fn
    #       when using estimator.train_and_evaluate, compute max_steps to set stop condition
    if option == 'simple':
        # train model
        mnist_classifier.train(
            input_fn=lambda: data_input_fn(train_dataset_fn_list, n_train_images, True, epochs, batch_size),
            hooks=None,
            steps=None)

        # evaluate the model and print results
        # hooks not working for evaluation?
        eval_results = mnist_classifier.evaluate(
            input_fn=lambda: data_input_fn(eval_dataset_fn_list, n_eval_images, False, 1, 1))
        print(eval_results)
    else:
        # create train_spec
        train_max_step = (n_train_images * epochs) // batch_size  # None: forever, default: None
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: data_input_fn(train_dataset_fn_list, n_train_images, True, 1, batch_size),
            max_steps=train_max_step
        )

        # create eval_spec
        eval_step = n_eval_images // batch_size  # None: forever, defalut: 100
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: data_input_fn(eval_dataset_fn_list, n_eval_images, False, 1, 1),
            steps=eval_step
        )

        # train & evaluate estimator
        # will save checkpoint per every epoch,
        # thus reporting summary for every epoch
        # will automatically add summary for loss, eval_metric, global step
        tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)
    return


def inference():
    # load mnist data
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    eval_images = mnist.test.images
    eval_labels = mnist.test.labels
    eval_labels = np.asarray(eval_labels, dtype=np.int32)

    # Load trained Estimator
    model_dir = './models'
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=model_dir)

    # predict
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': eval_images},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    # estimator's predict returns generator
    predicted_results = mnist_classifier.predict(input_fn=predict_input_fn)

    # need to iterate it and extract & print result
    for ii, results in enumerate(predicted_results):
        gt_label = eval_labels[ii]
        predicted_label = results['output_classes']
        print('gt - output: {} - {}'.format(gt_label, predicted_label))

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


def create_serving_model():
    # Load trained Estimator
    model_dir = './models'
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=model_dir)

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
    # 1. train the model
    train(fresh_training=True, option='utility_function')

    # # 2. test prediction with current saved model files
    # inference()
    #
    # # 3. make tensorflow serving files
    # create_serving_model()
    return


if __name__ == '__main__':
    main()
