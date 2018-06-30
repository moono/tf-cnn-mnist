import tensorflow as tf


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


def data_input_fn(data_fn, is_training, batch_size, epochs):
    dataset = tf.data.TFRecordDataset(data_fn)

    dataset = dataset.map(parse_tfrecord)

    if is_training:
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.prefetch(batch_size)
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    features = {
        'x': images,
    }
    return features, labels
