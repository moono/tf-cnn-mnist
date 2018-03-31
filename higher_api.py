import os
import shutil
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


# model_fn with tf.estimator.Estimator function signature
def cnn_model_fn(features, labels, mode, params):
    '''
    :param features: dictionary with single key 'x' which represents input images 
    :param labels: ground truth label
    :param mode: tensorflow mode - TRAIN, PREDICT, EVAL
    :param params: dictionay of additional parameter
    :return: tf.estimator.EstimatorSpec
    '''

    # ================================
    # common operations for all modes
    # ================================
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    inputs = tf.reshape(features['x'], shape=[-1, 28, 28, 1])

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
    logits = tf.layers.dense(dropout4, units=10)

    # ================================
    # prediction & serving mode
    # mode == tf.estimator.ModeKeys.PREDICT == 'infer'
    # ================================
    predictions = {
        'output_classes': tf.cast(tf.argmax(logits, axis=1), dtype=tf.int32, name='output_class'),
        'probabilities': tf.nn.softmax(logits, name='probs'),
    }
    # export output must be one of tf.estimator.export. ... class NOT a Tensor
    export_outputs = {
        'output_classes': tf.estimator.export.PredictOutput(predictions['output_classes']),
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # ================================
    # training mode
    # ================================
    onehot_labels = tf.one_hot(labels, depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_ops = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_ops)

    # ================================
    # evaluation mode
    # ================================
    # int32_labels = tf.cast(onehot_labels, dtype=tf.int32)
    # correct_prediction = tf.cast(tf.equal(int32_labels, predictions['output_classes']), dtype=tf.float32)
    eval_metric_ops = {
        # 'accuracy': tf.reduce_mean(correct_prediction),
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['output_classes'])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def train():
    # load mnist data
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    train_images = mnist.train.images
    train_labels = mnist.train.labels
    train_labels = np.asarray(train_labels, dtype=np.int32)
    eval_images = mnist.test.images
    eval_labels = mnist.test.labels
    eval_labels = np.asarray(eval_labels, dtype=np.int32)

    # hyper parameters
    batch_size = 100
    epochs = 20
    model_dir = './models'

    # clear saved model directory
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=model_dir)

    # # setup logging hook
    # tensors_to_log = {
    #     'probabilities': 'probs'
    # }
    # logging_hook = tf.train.LoggingTensorHook(tensors_to_log, every_n_iter=50)

    # train model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_images},
        y=train_labels,
        batch_size=batch_size,
        num_epochs=epochs,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        hooks=None,
        steps=None)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': eval_images},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
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
    train()

    # 2. test prediction with current saved model files
    inference()

    # 3. make tensorflow serving files
    create_serving_model()
    return


if __name__ == '__main__':
    main()
