import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def create_dummy_accuracy_test_data():
    n_data = 1000
    predicted_label = np.random.randint(0, 100, size=(n_data, 1), dtype=np.int32)
    ground_truth_label = np.random.randint(0, 100, size=(n_data, 1), dtype=np.int32)
    current_data_accuracy = (predicted_label == ground_truth_label).all(axis=1).mean(dtype=np.float32)
    return predicted_label, ground_truth_label, current_data_accuracy


def input_fn(inputs, labels, batch_size, epochs, is_training):
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    if is_training:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=1)

    iterator = dataset.make_one_shot_iterator()
    parsed_inputs, parsed_labels = iterator.get_next()
    features = {
        'inputs_1': parsed_inputs,
    }
    return features, parsed_labels


def accuracy_model_fn(features, labels, mode, params):
    dummy_inputs = features['inputs_1']

    # create dummy graph
    x = tf.cast(dummy_inputs, dtype=tf.float32)
    x = tf.layers.dense(x, units=100, activation=tf.nn.relu)
    logits = tf.layers.dense(x, units=1)
    dummy_predictions = tf.nn.sigmoid(logits)

    # bypass inputs as predicted classes
    predicted_classes = dummy_inputs

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'predicted_classes': predicted_classes,
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # set dummy loss value
    loss = tf.losses.mean_squared_error(labels=labels, predictions=dummy_predictions)

    # compute accuracy
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {
            'accuracy': accuracy,
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_ops = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_ops)


def main():
    # hyper parameters
    batch_size = 100
    epochs = 20

    # get data
    inputs, labels, acc = create_dummy_accuracy_test_data()
    print('Accuracy should be...: {}'.format(acc))

    # create the Estimator
    model_dir = '/tmp/check-accuracy'
    model = tf.estimator.Estimator(
        model_fn=accuracy_model_fn,
        model_dir=model_dir,
        config=None,
        params=None,
        warm_start_from=None
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(inputs, labels, batch_size, epochs, True),
        max_steps=1000
    )
    eval_spec = tf.estimator.EvalSpec(lambda: input_fn(inputs, labels, batch_size, epochs, False))

    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    return


if __name__ == '__main__':
    main()
