import os
import numpy as np
import tensorflow as tf


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/class/label': int64_feature(class_id),
    }))


def add_to_tfrecord(images, labels, tfrecord_writer):
    """
    :param images: [None, 784] float32 0.0 ~ 1.0 valued!!
    :param labels: [None] integer
    :param tfrecord_writer: wrtier object
    :return: None
    """
    # convert to desired shape
    desired_shape = (28, 28, 1)

    # parse numer of images
    num_images = images.shape[0]

    # start converting...
    with tf.Graph().as_default():
        image = tf.placeholder(dtype=tf.float32, shape=desired_shape)
        converted_dtype = tf.image.convert_image_dtype(image, dtype=tf.uint8)
        encoded_png = tf.image.encode_png(converted_dtype)

        with tf.Session('') as sess:
            for ii in range(num_images):
                print_idx = ii + 1
                if print_idx % 1000 == 0:
                    print('Converting image {:d}/{:d}'.format(print_idx, num_images))

                reshaped = np.reshape(images[ii], newshape=desired_shape)
                png_string = sess.run(encoded_png, feed_dict={image: reshaped})

                example = image_to_tfexample(png_string, labels[ii])
                tfrecord_writer.write(example.SerializeToString())


# def split_almost_even(a, n):
#     k, m = divmod(len(a), n)
#     return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def create_mnist_data():
    # set parameters
    n_split = 2
    output_dir = './data'

    # load mnist data
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')

    # create train examples
    train_images = mnist.train.images
    train_labels = mnist.train.labels
    train_labels = np.asarray(train_labels, dtype=np.int32)

    # split data
    splitted_train_images = np.array_split(train_images, n_split)
    splitted_train_labels = np.array_split(train_labels, n_split)
    for ii in range(n_split):
        tfrecord_fn = os.path.join(output_dir, 'mnist-train-{:02d}.tfrecord'.format(ii))
        with tf.python_io.TFRecordWriter(tfrecord_fn) as tfrecord_writer:
            add_to_tfrecord(splitted_train_images[ii], splitted_train_labels[ii], tfrecord_writer)

    # create validation examples
    eval_images = mnist.test.images
    eval_labels = mnist.test.labels
    eval_labels = np.asarray(eval_labels, dtype=np.int32)

    # split data
    splitted_eval_images = np.array_split(eval_images, n_split)
    splitted_eval_labels = np.array_split(eval_labels, n_split)
    for ii in range(n_split):
        tfrecord_fn = os.path.join(output_dir, 'mnist-val-{:02d}.tfrecord'.format(ii))
        with tf.python_io.TFRecordWriter(tfrecord_fn) as tfrecord_writer:
            add_to_tfrecord(splitted_eval_images[ii], splitted_eval_labels[ii], tfrecord_writer)
    return


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


def test_tfrecords():
    n_train = 55000
    n_eval = 10000
    epochs = 1
    batch_size = 1000
    filenames_tensor = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filenames_tensor)
    dataset = dataset.shuffle(buffer_size=n_train)
    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    training_fn_list = ['./data/mnist-train-00.tfrecord', './data/mnist-train-01.tfrecord']
    validate_fn_list = ['./data/mnist-val-00.tfrecord', './data/mnist-val-01.tfrecord']

    print('n_train: {:d}, n_eval: {:d}, epochs: {:d}'.format(n_train, n_eval, epochs))
    train_total_size = 0
    val_total_size = 0
    with tf.Session() as sess:
        sess.run(iterator.initializer, feed_dict={filenames_tensor: training_fn_list})

        while True:
            try:
                image, label = sess.run(next_element)
                train_total_size += label.shape[0]
            except tf.errors.OutOfRangeError:
                print('End of dataset')
                print('Train examples examined: {:d}'.format(train_total_size))
                break

        sess.run(iterator.initializer, feed_dict={filenames_tensor: validate_fn_list})

        while True:
            try:
                image, label = sess.run(next_element)
                val_total_size += label.shape[0]
            except tf.errors.OutOfRangeError:
                print('End of dataset')
                print('Eval examples examined: {:d}'.format(val_total_size))
                break

    # ===========================================
    # Expected output
    # ===========================================
    # n_train: 55000, n_eval: 10000, epochs: 1
    # End of dataset
    # Train examples examined: 55000
    # End of dataset
    # Eval examples examined: 10000
    #
    # n_train: 55000, n_eval: 10000, epochs: 2
    # End of dataset
    # Train examples examined: 110000
    # End of dataset
    # Eval examples examined: 20000
    return


def main():
    create_mnist_data()
    test_tfrecords()
    return


if __name__ == '__main__':
    main()
