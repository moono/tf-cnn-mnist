import os
import glob
import numpy as np
import tensorflow as tf


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def add_to_tfrecord(dummy, tfrecord_writer):
    # parse size
    n_data = dummy.shape[0]

    # start converting...
    with tf.Graph().as_default():
        for ii in range(n_data):
            example = tf.train.Example(features=tf.train.Features(feature={
                'dummy': int64_feature(dummy[ii]),
            }))
            tfrecord_writer.write(example.SerializeToString())


def create_dummy_data():
    # set parameters
    n_split = 100
    output_dir = './data'

    # load dummy data
    n_data = 1000
    dummy_data = [val for val in range(n_data)]
    dummy_data = np.array(dummy_data)

    # split data
    splitted_data = np.array_split(dummy_data, n_split)
    for ii in range(n_split):
        tfrecord_fn = os.path.join(output_dir, 'dummy-{:03d}.tfrecord'.format(ii))
        with tf.python_io.TFRecordWriter(tfrecord_fn) as tfrecord_writer:
            add_to_tfrecord(splitted_data[ii], tfrecord_writer)
    return


def parse_tfrecord(raw_record):
    keys_to_features = {
        'dummy': tf.FixedLenFeature((), tf.int64),
    }

    # parse feature
    parsed = tf.parse_single_example(raw_record, keys_to_features)
    label = tf.cast(parsed['dummy'], tf.int32)
    return label


def test_tfrecords(case):
    n_elements_per_file = 10
    epochs = 1
    batch_size = 10

    # prepare tfrecords
    # sort the files so that we can understand how dataset api work accordingly...
    tfrecord_dir = './data'
    dummy_fn_list = glob.glob(os.path.join(tfrecord_dir, 'dummy-*.tfrecord'))
    dummy_fn_list = sorted(dummy_fn_list)

    # start dataset with unshuffled tfrecord file list (default: shuffle=True, added since r1.7)
    dataset = tf.data.Dataset.list_files(dummy_fn_list, shuffle=False)

    # 1:
    # no shuffling tfrecord files
    # cycle_length=1
    # no shuffling data elements
    if case == 1:
        dataset = dataset.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=1))

    # 2:
    # no shuffling tfrecord files
    # cycle_length=2
    # no shuffling data elements
    elif case == 2:
        dataset = dataset.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=2))

    # 3:
    # shuffling tfrecord files (buffer_size=number of files)
    # cycle_length=1
    # no shuffling data elements
    elif case == 3:
        dataset = dataset.shuffle(len(dummy_fn_list))
        dataset = dataset.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=1))

    # 4:
    # shuffling tfrecord files (buffer_size=number of files)
    # cycle_length=2
    # no shuffling data elements
    elif case == 4:
        dataset = dataset.shuffle(len(dummy_fn_list))
        dataset = dataset.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=2))

    # 5:
    # no shuffling tfrecord files
    # cycle_length=1
    # shuffling data elements with buffer size larger than single tfrecord's element size
    elif case == 5:
        dataset = dataset.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=1))
        dataset = dataset.shuffle(buffer_size=n_elements_per_file + 1)

    # 6:
    # no shuffling tfrecord files
    # cycle_length=1
    # shuffling data elements with buffer size smaller than single tfrecord's element size
    elif case == 6:
        dataset = dataset.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=1))
        dataset = dataset.shuffle(buffer_size=2)

    # 7:
    # no shuffling tfrecord files
    # cycle_length=2
    # shuffling data elements with buffer size larger than single tfrecord's element size
    elif case == 7:
        dataset = dataset.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=2))
        dataset = dataset.shuffle(buffer_size=n_elements_per_file + 1)

    # 8:
    # shuffling tfrecord files (buffer_size=number of files)
    # cycle_length=1
    # shuffling data elements with buffer size larger than single tfrecord's element size
    elif case == 8:
        dataset = dataset.shuffle(len(dummy_fn_list))
        dataset = dataset.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=1))
        dataset = dataset.shuffle(buffer_size=n_elements_per_file + 1)

    # 9:
    # shuffling tfrecord files (buffer_size=number of files)
    # cycle_length=1
    # shuffling data elements with buffer size smaller than single tfrecord's element size
    elif case == 9:
        dataset = dataset.shuffle(len(dummy_fn_list))
        dataset = dataset.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=1))
        dataset = dataset.shuffle(buffer_size=2)

    # 10: I think this is recommended way for proper shuffling with large dataset
    # shuffling tfrecord files (buffer_size=number of files)
    # cycle_length=2 ==> 1 < cycle_length < element shuffle's buffer size
    # shuffling data elements with buffer size larger than single tfrecord's element size
    elif case == 10:
        dataset = dataset.shuffle(len(dummy_fn_list))
        dataset = dataset.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=2))
        dataset = dataset.shuffle(buffer_size=n_elements_per_file + 1)
    else:
        raise ValueError('test case must be [1~10]')

    dataset = dataset.map(parse_tfrecord, num_parallel_calls=4)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    output_fn = 'case-{:03d}.txt'.format(case)
    text_file = open(output_fn, 'w')

    with tf.Session() as sess:
        while True:
            try:
                labels = sess.run(next_element)
                print('{}'.format(labels))
                text_file.write('{}\n'.format(labels))
            except tf.errors.OutOfRangeError:
                print('End of dataset')
                break

    text_file.close()
    return


def main():
    # # Will create 100 tfrecords files each containing 10 labels
    # # [0~9], [10~19], [20~29], ..., ..., [980~989], [990~999]
    # create_dummy_data()

    n_test_case = 10
    for ii in range(n_test_case):
        t_case = ii + 1
        test_tfrecords(case=t_case)
    return


if __name__ == '__main__':
    main()
