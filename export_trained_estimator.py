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
        'exported_output': tf.estimator.export.PredictOutput(
            {
                'dense4': tf.identity(dense4, name='dense4'),
                'output_classes': predictions['class_id'],
            }
        ),
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
        model_dir=model_dir,
        config=None,
        params={
            'input_size': 28,
            'n_output_classes': 10,
        },
        warm_start_from=model_dir,
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
    # 3. make tensorflow serving files
    create_serving_model()
    return


if __name__ == '__main__':
    main()
