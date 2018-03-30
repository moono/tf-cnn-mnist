import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


# tf.estimator.Estimator function signature
def cnn_model_fn(features, labels, mode, params):
    '''
    :param features: dictionary with single key 'x' which represents input images 
    :param labels: 
    :param mode: tensorflow mode - TRAIN, PREDICT, EVAL, INFER
    :param params: dictionay of additional parameter
    :return: 
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
    # prediction mode
    # ================================
    predictions = {
        'classes': tf.cast(tf.argmax(logits, axis=1), dtype=tf.int32),
        'probabilities': tf.nn.softmax(logits, name='probs'),
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

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
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
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


if __name__ == '__main__':
    main()
