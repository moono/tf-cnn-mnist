import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


tf.logging.set_verbosity(tf.logging.INFO)


def generate_dataset():
    x_batch = np.linspace(0, 2, 100)
    y_batch = 1.5 * x_batch + np.random.randn(*x_batch.shape) * 0.2 + 0.5
    return x_batch, y_batch


def simple_network(x, w):
    with tf.variable_scope('lreg', reuse=tf.AUTO_REUSE):
        b = tf.get_variable('b', shape=[], dtype=tf.float32, initializer=tf.random_normal_initializer())
        y_pred = tf.add(tf.multiply(w, x), b)
    return y_pred


def run():
    # prepare data
    x_batch, y_batch = generate_dataset()

    # prepare placeholders
    x = tf.placeholder(tf.float32, shape=(None,), name='x')
    y = tf.placeholder(tf.float32, shape=(None,), name='y')

    # construct network
    # w = tf.get_variable('w', shape=[], dtype=tf.float32, initializer=tf.initializers.ones())
    w = tf.get_variable('w', shape=[], dtype=tf.float32, initializer=tf.random_normal_initializer())
    y_pred = simple_network(x, w)

    # prepare exponential moving average
    ema = tf.train.ExponentialMovingAverage(decay=0.998)
    ema_op = ema.apply([w])
    ema_w = ema.average(w)

    # eval network
    y_pred_eval = simple_network(x, ema_w)

    # compute loss
    loss = tf.reduce_mean(tf.square(y_pred - y))

    # prepare optimizer op
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    with tf.control_dependencies([ema_op]):
        train_op = optimizer.minimize(loss)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        feed_dict = {x: x_batch, y: y_batch}

        for i in range(300):
            _, loss_val = session.run([train_op, loss], feed_dict)
            w_o, w_a = session.run([w, ema_w], feed_dict)
            print('{}: loss {}, w {}, w_av: {}'.format(i, loss_val, w_o, w_a))

        print('Predicting')
        y_pred_batch = session.run(y_pred, {x: x_batch})
        y_pred_batch_eval = session.run(y_pred_eval, {x: x_batch})

    plt.scatter(x_batch, y_batch)
    plt.plot(x_batch, y_pred_batch, color='red')
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.savefig('plot.png')
    plt.close()

    plt.scatter(x_batch, y_batch)
    plt.plot(x_batch, y_pred_batch_eval, color='red')
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.savefig('plot-eval.png')
    plt.close()


def input_fn(x_batch, y_batch, epoch, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x_batch, y_batch))

    dataset = dataset.shuffle(buffer_size=20000).repeat(epoch)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(
        map_func=lambda x, y: (
            {
                'x': tf.cast(x, dtype=tf.float32),
            }, tf.cast(y, dtype=tf.float32)),
        num_parallel_calls=8
    )
    return dataset


def model_fn(features, labels, mode, params):
    # ================================
    # common operations for all modes
    # ================================
    x = features['x']
    y = labels

    is_training = mode == tf.estimator.ModeKeys.TRAIN

    # construct network
    # w = tf.get_variable('w', shape=[], dtype=tf.float32, initializer=tf.initializers.ones())
    w = tf.get_variable('w', shape=[], dtype=tf.float32, initializer=tf.random_normal_initializer())
    y_pred = simple_network(x, w)

    # apply exponential moving average
    ema = tf.train.ExponentialMovingAverage(decay=0.998)
    ema_op = ema.apply([w])
    w_average = ema.average(w)

    # eval network
    y_pred_eval = simple_network(x, w_average)

    # ================================
    # prediction mode
    # ================================
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={})

    # compute loss
    if is_training:
        loss = tf.reduce_mean(tf.square(y_pred - y))
    else:
        loss = tf.reduce_mean(tf.square(y_pred_eval - y))

    tf.summary.scalar('w_ori', w)
    tf.summary.scalar('w_avg', w_average)

    # ================================
    # evaluation mode
    # ================================
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={})

    # ================================
    # training mode
    # ================================
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    with tf.control_dependencies([ema_op]):
        train_ops = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_ops)


def train_estimator(model_dir):
    # prepare data
    x_batch, y_batch = generate_dataset()

    epoch = 300
    batch_size = 1

    # create run config for estimator
    run_config = tf.estimator.RunConfig(keep_checkpoint_max=1)

    # create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        config=run_config,
        params={},
        warm_start_from=None
    )

    # train model
    mnist_classifier.train(input_fn=lambda: input_fn(x_batch, y_batch, epoch, batch_size))

    # evaluate the model and print results
    # hooks not working for evaluation?
    eval_results = mnist_classifier.evaluate(input_fn=lambda: input_fn(x_batch, y_batch, 1, batch_size))
    print(eval_results)

    return


def main():
    # run()

    model_dir = './models/ema_regression'
    train_estimator(model_dir)
    return


if __name__ == "__main__":
    main()