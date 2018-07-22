import os
import glob
import math
import pickle
import numpy as np
import tensorflow as tf


def create_sample_data(output_dir):
    x_dir = os.path.join(output_dir, 'x')
    y_dir = os.path.join(output_dir, 'y')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(x_dir):
        os.makedirs(x_dir)
    if not os.path.exists(y_dir):
        os.makedirs(y_dir)

    # pick 10000 samples
    # x = [100000, 100001, ..., 19999]
    # y = [200000, 200001, ..., 29999]
    for ii in range(10000):
        x = np.ones(shape=(1,), dtype=np.int32) * ii + 10000
        y = np.ones(shape=(1,), dtype=np.int32) * ii + 20000

        with open(os.path.join(x_dir, '{:08d}.pkl'.format(ii)), 'wb') as f:
            pickle.dump(x, f)
        with open(os.path.join(y_dir, '{:08d}.pkl'.format(ii)), 'wb') as f:
            pickle.dump(y, f)
    return


def parse_pkl_fn(x_fn, y_fn):
    with open(x_fn, 'rb') as f:
        x = pickle.load(f)
    with open(y_fn, 'rb') as f:
        y = pickle.load(f)
    return x, y


def parse_fn(x, y):
    x = tf.cast(x, dtype=tf.int32)
    y = tf.cast(y, dtype=tf.int32)

    x.set_shape([1])
    y.set_shape([1])
    return x, y


def input_fn(x_fns, y_fns, batch_size, epochs, pre_shuffle, post_shuffle, pre_shuffle_size, post_shuffle_size):
    dataset = tf.data.Dataset.from_tensor_slices((x_fns, y_fns))
    print(dataset)

    if pre_shuffle:
        dataset = dataset.shuffle(buffer_size=pre_shuffle_size, seed=777)
        print(dataset)

    dataset = dataset.map(lambda x_fn, y_fn: tf.py_func(
        parse_pkl_fn,
        [x_fn, y_fn],
        [tf.int32, tf.int32])
    )
    print(dataset)

    dataset = dataset.map(parse_fn)
    print(dataset)

    if post_shuffle:
        dataset = dataset.shuffle(buffer_size=post_shuffle_size, seed=777)
        print(dataset)

    dataset = dataset.prefetch(batch_size)
    print(dataset)
    dataset = dataset.batch(batch_size)
    print(dataset)
    dataset = dataset.repeat(epochs)
    print(dataset)

    iterator = dataset.make_one_shot_iterator()
    x, y = iterator.get_next()

    features = {
        'x': x,
    }
    return features, y


def test_input_fn(data_dir):
    # collect files
    x_fns = glob.glob(os.path.join(data_dir, 'x', '*.pkl'))
    y_fns = glob.glob(os.path.join(data_dir, 'y', '*.pkl'))
    x_fns = sorted(x_fns)
    y_fns = sorted(y_fns)

    n_samples = len(x_fns)
    batch_size = 7
    epochs = 2
    pre_shuffle = True
    post_shuffle = False
    pre_shuffle_size = n_samples
    # pre_shuffle_size = 1000
    post_shuffle_size = batch_size
    # post_shuffle_size = n_samples
    test_x, test_y = input_fn(x_fns, y_fns, batch_size, epochs,
                              pre_shuffle, post_shuffle, pre_shuffle_size, post_shuffle_size)

    estimated_steps = math.ceil(n_samples / batch_size) * epochs
    steps_per_epoch = estimated_steps / epochs
    counts = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print('Number of test samples: {}'.format(n_samples))
        print('Batch size: {}'.format(batch_size))
        print('Epoch size: {}'.format(epochs))
        print('Pre Shuffle: {} with {}'.format(pre_shuffle, pre_shuffle_size))
        print('Post Shuffle: {} with {}'.format(post_shuffle, post_shuffle_size))
        print('Estimated steps: {}'.format(estimated_steps))
        print('  x    y')
        while True:
            try:
                x, y = sess.run([test_x, test_y])

                residuals = counts % steps_per_epoch
                if residuals == 0 or residuals == steps_per_epoch - 1:
                    x = x['x']
                    xy = np.concatenate((x, y), axis=1)
                    print(xy)
                    print('-------------')

                counts += 1
            except tf.errors.OutOfRangeError:
                print('End of dataset')
                break
    return


def main():
    data_path_base = './data'
    data_dir = os.path.join(data_path_base, 'dataset_api_test_simple')

    # 1. create sample data
    create_sample_data(data_dir)

    # 2. load sample data & play with it
    test_input_fn(data_dir)
    return


if __name__ == '__main__':
    main()
