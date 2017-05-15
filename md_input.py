# ==============================================================================

"""Base utilities for loading datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
from os import path
import random
import tempfile
import time

import numpy as np
from datetime import datetime
from tensorflow.python.framework import dtypes
from six.moves import urllib
import tensorflow as tf
from tensorflow.contrib.framework import deprecated
from tensorflow.python.platform import gfile

Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
Datasets_train = collections.namedtuple('Datasets', ['train', 'validation'])
Datasets_test = collections.namedtuple('Datasets', ['test', 'test0',
                                                    'test1', 'test2', 'test24',
                                                    'test25', 'test26', 'test27',
                                                    'test28', 'test29',
                                                    'test30', 'test31', 'test32',
                                                    'test33', 'test34', 'test35', 'test36'])
Datasets_multi_testing_set = collections.namedtuple('Datasets', ['train', 'validation', 'test', 'test0',
                                                                 'test1', 'test2', 'test24',
                                                                 'test25', 'test26', 'test27',
                                                                 'test28', 'test29',
                                                                 'test30', 'test31', 'test32',
                                                                 'test33', 'test34', 'test35', 'test36'])
VIDEO_TRAINING = "/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/train_data_32x32/training_32x32_equal.csv"
# VIDEO_TRAINING = "/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/training_data_4_fake_without_comma.csv"
VIDEO_TESTING = "/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32.csv"
VIDEO_TESTING0 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_0.csv'
VIDEO_TESTING1 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_1.csv'
VIDEO_TESTING2 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_2.csv'
VIDEO_TESTING3 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_3.csv'
VIDEO_TESTING4 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_4.csv'
VIDEO_TESTING5 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_5.csv'
VIDEO_TESTING6 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_6.csv'
VIDEO_TESTING7 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_7.csv'
VIDEO_TESTING8 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_8.csv'
VIDEO_TESTING9 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_9.csv'
VIDEO_TESTING10 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_10.csv'
VIDEO_TESTING11 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_11.csv'
VIDEO_TESTING12 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_12.csv'
VIDEO_TESTING13 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_13.csv'
VIDEO_TESTING14 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_14.csv'
VIDEO_TESTING15 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_15.csv'
VIDEO_TESTING16 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_16.csv'
VIDEO_TESTING17 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_17.csv'
VIDEO_TESTING18 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_18.csv'
VIDEO_TESTING19 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_19.csv'
VIDEO_TESTING20 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_20.csv'
VIDEO_TESTING21 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_21.csv'
VIDEO_TESTING22 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_22.csv'
VIDEO_TESTING23 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_23.csv'
VIDEO_TESTING24 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_24.csv'
VIDEO_TESTING25 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_25.csv'
VIDEO_TESTING26 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_26.csv'
VIDEO_TESTING27 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_27.csv'
VIDEO_TESTING28 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_28.csv'
VIDEO_TESTING29 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_29.csv'
VIDEO_TESTING30 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_30.csv'
VIDEO_TESTING31 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_31.csv'
VIDEO_TESTING32 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_32.csv'
VIDEO_TESTING33 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_33.csv'
VIDEO_TESTING34 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_34.csv'
VIDEO_TESTING35 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_35.csv'
VIDEO_TESTING36 = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32_36.csv'


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    print("number of labels : " + str(num_labels))
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    print("type of labels_one_hot :     " + str(type(labels_one_hot)))
    print("shape of labels_one_hot :     " + str(labels_one_hot.shape))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

    # num_labels = labels_dense.shape[0]
    # index_offset = numpy.arange(num_labels) * num_classes
    # labels_one_hot = numpy.zeros((num_labels, num_classes))
    # labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    # return labels_one_hot


def load_csv_without_header(filename,
                            target_dtype,
                            features_dtype,
                            n_samples,
                            block_size=32,
                            target_column=-1,
                            ):
    """Load dataset from CSV file with a header row."""
    with gfile.Open(filename) as csv_file:
        data_file = csv.reader(csv_file)
        # header = next(data_file)
        n_samples = n_samples
        # n_samples = 5124

        n_features = block_size ** 2
        data = np.zeros((n_samples, n_features), dtype=features_dtype)
        target = np.zeros((n_samples,), dtype=target_dtype)
        for i, row in enumerate(data_file):
            qwer = np.asarray(row.pop(0), dtype=target_dtype)
            target[i] = np.asarray(row.pop(target_column), dtype=target_dtype)
            data[i] = np.asarray(row, dtype=features_dtype)

    # print(type(data))
    # print(data)
    # print(data.ndim)
    # print("==============================")
    # print("============================== now flatten")
    data = data.flatten()
    data = data.reshape(n_samples, block_size, block_size, 1)
    # target = target.flatten()
    # target = target.reshape(n_samples, block_size, block_size, 1)
    target = dense_to_one_hot(target, 37)

    # print(type(data))
    # print(data)
    # print(len(data))
    # print(data.ndim)
    return Dataset(data=data, target=target)


class DataSet(object):
    def __init__(self,
                 images,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float32,
                 reshape=True):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            self._num_examples = images.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            if reshape:
                assert images.shape[3] == 1
                images = images.reshape(images.shape[0],
                                        images.shape[1] * images.shape[2])
            if dtype == dtypes.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                images = images.astype(np.float32)
                images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_data_sets(dtype=dtypes.float32,
                   reshape=False,
                   validation_size=0):
    start_time_train_data_reading = datetime.now()
    print(str(start_time_train_data_reading) + " >>>>-------> start reading the training csv file.")
    r = csv.reader(open(VIDEO_TRAINING))  # Here your csv file

    end_for_reading_training_csv = datetime.now()
    duration_reading_training_csv = end_for_reading_training_csv - start_time_train_data_reading
    print("duration for reading training csv: " + str(duration_reading_training_csv))

    # r = csv.reader(open(VIDEO_TESTING1))  # Here your csv file
    row_count = sum(1 for _ in r)
    print("row_count of the training csv file: " + str(row_count))

    end_for_row_count_of_training_csv = datetime.now()
    duration_row_count_training_csv = end_for_row_count_of_training_csv - end_for_reading_training_csv
    print("duration for row counting of training csv: " + str(duration_row_count_training_csv))

    # training set start-------------------------------------------------------->
    # train_set = load_csv_without_header(filename=VIDEO_TESTING1, target_dtype=np.int, features_dtype=np.int,
    train_set = load_csv_without_header(filename=VIDEO_TRAINING, target_dtype=np.int, features_dtype=np.int,
                                        n_samples=row_count)  # 4328116
    end_of_loading_csv_without_header = datetime.now()
    duration_train_data_reading = end_of_loading_csv_without_header - end_for_row_count_of_training_csv
    print(str(end_of_loading_csv_without_header) + "   ***-------> now end reading the training csv file.")
    print("duration for loading_csv_without_header: " + str(duration_train_data_reading))
    # training set end -------------------------------------------------------->

    # n_samples=51, block_size=4)  # 4328116

    train_images = train_set.data
    train_labels = train_set.target

    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 1.")
    r = csv.reader(open(VIDEO_TESTING))  # Here your csv file
    row_count = sum(1 for _ in r)

    test_set = load_csv_without_header(
        filename=VIDEO_TESTING,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count,
    )
    test_images = test_set.data
    test_labels = test_set.target
    # testing set end -------------------------------------------------------->

    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 2.")
    r = csv.reader(open(VIDEO_TESTING0))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set0 = load_csv_without_header(
        filename=VIDEO_TESTING0,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count,
    )
    test_images0 = test_set0.data
    test_labels0 = test_set0.target

    # testing set end -------------------------------------------------------->

    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 3.")
    r = csv.reader(open(VIDEO_TESTING1))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set1 = load_csv_without_header(
        filename=VIDEO_TESTING1,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count,
    )
    test_images1 = test_set1.data
    test_labels1 = test_set1.target

    # testing set end -------------------------------------------------------->

    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 4.")
    r = csv.reader(open(VIDEO_TESTING2))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set2 = load_csv_without_header(
        filename=VIDEO_TESTING2,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count,
    )
    test_images2 = test_set2.data
    test_labels2 = test_set2.target

    # testing set end -------------------------------------------------------->

    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 5.")
    r = csv.reader(open(VIDEO_TESTING24))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set24 = load_csv_without_header(
        filename=VIDEO_TESTING24,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count,
    )
    test_images24 = test_set24.data
    test_labels24 = test_set24.target

    # testing set end -------------------------------------------------------->
    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 6.")
    r = csv.reader(open(VIDEO_TESTING25))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set25 = load_csv_without_header(
        filename=VIDEO_TESTING25,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count,
    )
    test_images25 = test_set25.data
    test_labels25 = test_set25.target

    # testing set end -------------------------------------------------------->
    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 7.")
    r = csv.reader(open(VIDEO_TESTING26))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set26 = load_csv_without_header(
        filename=VIDEO_TESTING26,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images26 = test_set26.data
    test_labels26 = test_set26.target

    # testing set end -------------------------------------------------------->
    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 8.")
    r = csv.reader(open(VIDEO_TESTING27))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set27 = load_csv_without_header(
        filename=VIDEO_TESTING27,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images27 = test_set27.data
    test_labels27 = test_set27.target
    # testing set end -------------------------------------------------------->

    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 9.")
    r = csv.reader(open(VIDEO_TESTING28))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set28 = load_csv_without_header(
        filename=VIDEO_TESTING28,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images28 = test_set28.data
    test_labels28 = test_set28.target
    # testing set end -------------------------------------------------------->

    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 10.")
    r = csv.reader(open(VIDEO_TESTING29))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set29 = load_csv_without_header(
        filename=VIDEO_TESTING29,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images29 = test_set29.data
    test_labels29 = test_set29.target
    # testing set end -------------------------------------------------------->

    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 11.")
    r = csv.reader(open(VIDEO_TESTING30))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set30 = load_csv_without_header(
        filename=VIDEO_TESTING30,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images30 = test_set30.data
    test_labels30 = test_set30.target
    # testing set end -------------------------------------------------------->

    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 12.")
    r = csv.reader(open(VIDEO_TESTING31))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set31 = load_csv_without_header(
        filename=VIDEO_TESTING31,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images31 = test_set31.data
    test_labels31 = test_set31.target
    # testing set end -------------------------------------------------------->

    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 13.")
    r = csv.reader(open(VIDEO_TESTING32))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set32 = load_csv_without_header(
        filename=VIDEO_TESTING32,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images32 = test_set32.data
    test_labels32 = test_set32.target
    # testing set end -------------------------------------------------------->

    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 14.")
    r = csv.reader(open(VIDEO_TESTING33))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set33 = load_csv_without_header(
        filename=VIDEO_TESTING33,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images33 = test_set33.data
    test_labels33 = test_set33.target
    # testing set end -------------------------------------------------------->

    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 15.")
    r = csv.reader(open(VIDEO_TESTING34))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set34 = load_csv_without_header(
        filename=VIDEO_TESTING34,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images34 = test_set34.data
    test_labels34 = test_set34.target
    # testing set end -------------------------------------------------------->

    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 16.")
    r = csv.reader(open(VIDEO_TESTING35))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set35 = load_csv_without_header(
        filename=VIDEO_TESTING35,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images35 = test_set35.data
    test_labels35 = test_set35.target
    # testing set end -------------------------------------------------------->

    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 17.")
    r = csv.reader(open(VIDEO_TESTING36))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set36 = load_csv_without_header(
        filename=VIDEO_TESTING36,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images36 = test_set36.data
    test_labels36 = test_set36.target
    # testing set end -------------------------------------------------------->

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'.format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
    validation = DataSet(validation_images,
                         validation_labels,
                         dtype=dtype,
                         reshape=reshape)

    test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)
    test0 = DataSet(test_images0, test_labels0, dtype=dtype, reshape=reshape)
    test1 = DataSet(test_images1, test_labels1, dtype=dtype, reshape=reshape)
    test2 = DataSet(test_images2, test_labels2, dtype=dtype, reshape=reshape)
    test24 = DataSet(test_images24, test_labels24, dtype=dtype, reshape=reshape)
    test25 = DataSet(test_images25, test_labels25, dtype=dtype, reshape=reshape)
    test26 = DataSet(test_images26, test_labels26, dtype=dtype, reshape=reshape)
    test27 = DataSet(test_images27, test_labels27, dtype=dtype, reshape=reshape)
    test28 = DataSet(test_images28, test_labels28, dtype=dtype, reshape=reshape)
    test29 = DataSet(test_images29, test_labels29, dtype=dtype, reshape=reshape)
    test30 = DataSet(test_images30, test_labels30, dtype=dtype, reshape=reshape)
    test31 = DataSet(test_images31, test_labels31, dtype=dtype, reshape=reshape)
    test32 = DataSet(test_images32, test_labels32, dtype=dtype, reshape=reshape)
    test33 = DataSet(test_images33, test_labels33, dtype=dtype, reshape=reshape)
    test34 = DataSet(test_images34, test_labels34, dtype=dtype, reshape=reshape)
    test35 = DataSet(test_images35, test_labels35, dtype=dtype, reshape=reshape)
    test36 = DataSet(test_images36, test_labels36, dtype=dtype, reshape=reshape)

    end_time_for_reading_data = datetime.now()

    time_cost_for_reading_data = end_time_for_reading_data - start_time_train_data_reading

    print("")
    print("")
    print("****************************************************************")
    print("------")
    print("time cost for reading data : " + str(time_cost_for_reading_data))
    print("------")
    print("****************************************************************")

    return Datasets_multi_testing_set(train=train, validation=validation, test=test, test0=test0, test1=test1,
                                      test2=test2, test24=test24, test25=test25, test26=test26, test27=test27,
                                      test28=test28, test29=test29, test30=test30, test31=test31, test32=test32,
                                      test33=test33, test34=test34, test35=test35, test36=test36)


def read_train_data_sets(dtype=dtypes.float32,
                         reshape=False,
                         validation_size=0):
    BLOCK_SIZE = 32

    start_time_train_data_reading = datetime.now()
    print(str(start_time_train_data_reading) + " >>>>-------> start reading the training csv file.")
    r = csv.reader(open(VIDEO_TRAINING))  # Here your csv file

    end_for_reading_training_csv = datetime.now()
    duration_reading_training_csv = end_for_reading_training_csv - start_time_train_data_reading
    print("duration for reading training csv: " + str(duration_reading_training_csv))

    # r = csv.reader(open(VIDEO_TESTING1))  # Here your csv file
    row_count = sum(1 for _ in r)
    print("row_count of the training csv file: " + str(row_count))

    end_for_row_count_of_training_csv = datetime.now()
    duration_row_count_training_csv = end_for_row_count_of_training_csv - end_for_reading_training_csv
    print("duration for row counting of training csv: " + str(duration_row_count_training_csv))

    # training set start-------------------------------------------------------->
    # train_set = load_csv_without_header(filename=VIDEO_TESTING1, target_dtype=np.int, features_dtype=np.int,
    train_set = load_csv_without_header(filename=VIDEO_TRAINING, target_dtype=np.int, features_dtype=np.int,
                                        n_samples=row_count, block_size=BLOCK_SIZE)  # 4328116
    end_of_loading_csv_without_header = datetime.now()
    duration_train_data_reading = end_of_loading_csv_without_header - end_for_row_count_of_training_csv
    print(str(end_of_loading_csv_without_header) + "   ***-------> now end reading the training csv file.")
    print("duration for loading_csv_without_header: " + str(duration_train_data_reading))
    # training set end -------------------------------------------------------->

    # n_samples=51, block_size=4)  # 4328116

    train_images = train_set.data
    train_labels = train_set.target

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
                .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
    validation = DataSet(validation_images,
                         validation_labels,
                         dtype=dtype,
                         reshape=reshape)

    end_time_for_reading_data = datetime.now()

    time_cost_for_reading_data = end_time_for_reading_data - start_time_train_data_reading

    print("")
    print("")
    print("****************************************************************")
    print("------")
    print("time cost for reading data : " + str(time_cost_for_reading_data))
    print("------")
    print("****************************************************************")

    # train=train.next_batch(100)

    return Datasets_train(train=train, validation=validation)


def read_test_data_sets(dtype=dtypes.float32,
                        reshape=False,
                        ):
    start_time_train_data_reading = datetime.now()
    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 1.")
    r = csv.reader(open(VIDEO_TESTING))  # Here your csv file
    row_count = sum(1 for _ in r)

    test_set = load_csv_without_header(
        filename=VIDEO_TESTING,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images = test_set.data
    test_labels = test_set.target
    # testing set end -------------------------------------------------------->

    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 2.")
    r = csv.reader(open(VIDEO_TESTING0))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set0 = load_csv_without_header(
        filename=VIDEO_TESTING0,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images0 = test_set0.data
    test_labels0 = test_set0.target

    # testing set end -------------------------------------------------------->

    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 3.")
    r = csv.reader(open(VIDEO_TESTING1))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set1 = load_csv_without_header(
        filename=VIDEO_TESTING1,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images1 = test_set1.data
    test_labels1 = test_set1.target

    # testing set end -------------------------------------------------------->

    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 4.")
    r = csv.reader(open(VIDEO_TESTING2))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set2 = load_csv_without_header(
        filename=VIDEO_TESTING2,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images2 = test_set2.data
    test_labels2 = test_set2.target

    # testing set end -------------------------------------------------------->

    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 5.")
    r = csv.reader(open(VIDEO_TESTING24))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set24 = load_csv_without_header(
        filename=VIDEO_TESTING24,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images24 = test_set24.data
    test_labels24 = test_set24.target

    # testing set end -------------------------------------------------------->
    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 6.")
    r = csv.reader(open(VIDEO_TESTING25))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set25 = load_csv_without_header(
        filename=VIDEO_TESTING25,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images25 = test_set25.data
    test_labels25 = test_set25.target

    # testing set end -------------------------------------------------------->
    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 7.")
    r = csv.reader(open(VIDEO_TESTING26))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set26 = load_csv_without_header(
        filename=VIDEO_TESTING26,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images26 = test_set26.data
    test_labels26 = test_set26.target

    # testing set end -------------------------------------------------------->
    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 8.")
    r = csv.reader(open(VIDEO_TESTING27))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set27 = load_csv_without_header(
        filename=VIDEO_TESTING27,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images27 = test_set27.data
    test_labels27 = test_set27.target
    # testing set end -------------------------------------------------------->

    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 9.")
    r = csv.reader(open(VIDEO_TESTING28))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set28 = load_csv_without_header(
        filename=VIDEO_TESTING28,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images28 = test_set28.data
    test_labels28 = test_set28.target
    # testing set end -------------------------------------------------------->

    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 10.")
    r = csv.reader(open(VIDEO_TESTING29))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set29 = load_csv_without_header(
        filename=VIDEO_TESTING29,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images29 = test_set29.data
    test_labels29 = test_set29.target
    # testing set end -------------------------------------------------------->

    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 11.")
    r = csv.reader(open(VIDEO_TESTING30))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set30 = load_csv_without_header(
        filename=VIDEO_TESTING30,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images30 = test_set30.data
    test_labels30 = test_set30.target
    # testing set end -------------------------------------------------------->

    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 12.")
    r = csv.reader(open(VIDEO_TESTING31))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set31 = load_csv_without_header(
        filename=VIDEO_TESTING31,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images31 = test_set31.data
    test_labels31 = test_set31.target
    # testing set end -------------------------------------------------------->

    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 13.")
    r = csv.reader(open(VIDEO_TESTING32))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set32 = load_csv_without_header(
        filename=VIDEO_TESTING32,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images32 = test_set32.data
    test_labels32 = test_set32.target
    # testing set end -------------------------------------------------------->

    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 14.")
    r = csv.reader(open(VIDEO_TESTING33))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set33 = load_csv_without_header(
        filename=VIDEO_TESTING33,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images33 = test_set33.data
    test_labels33 = test_set33.target
    # testing set end -------------------------------------------------------->

    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 15.")
    r = csv.reader(open(VIDEO_TESTING34))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set34 = load_csv_without_header(
        filename=VIDEO_TESTING34,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images34 = test_set34.data
    test_labels34 = test_set34.target
    # testing set end -------------------------------------------------------->

    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 16.")
    r = csv.reader(open(VIDEO_TESTING35))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set35 = load_csv_without_header(
        filename=VIDEO_TESTING35,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images35 = test_set35.data
    test_labels35 = test_set35.target
    # testing set end -------------------------------------------------------->

    # testing set start -------------------------------------------------------->
    start_time_test_data_reading = datetime.now()
    print(str(start_time_test_data_reading) + "   ***-------> now start reading the testing csv file 17.")
    r = csv.reader(open(VIDEO_TESTING36))  # Here your csv file
    row_count = sum(1 for _ in r)
    test_set36 = load_csv_without_header(
        filename=VIDEO_TESTING36,
        target_dtype=np.int,
        features_dtype=np.int,
        n_samples=row_count
    )
    test_images36 = test_set36.data
    test_labels36 = test_set36.target
    # testing set end -------------------------------------------------------->

    test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)
    test0 = DataSet(test_images0, test_labels0, dtype=dtype, reshape=reshape)
    test1 = DataSet(test_images1, test_labels1, dtype=dtype, reshape=reshape)
    test2 = DataSet(test_images2, test_labels2, dtype=dtype, reshape=reshape)
    test24 = DataSet(test_images24, test_labels24, dtype=dtype, reshape=reshape)
    test25 = DataSet(test_images25, test_labels25, dtype=dtype, reshape=reshape)
    test26 = DataSet(test_images26, test_labels26, dtype=dtype, reshape=reshape)
    test27 = DataSet(test_images27, test_labels27, dtype=dtype, reshape=reshape)
    test28 = DataSet(test_images28, test_labels28, dtype=dtype, reshape=reshape)
    test29 = DataSet(test_images29, test_labels29, dtype=dtype, reshape=reshape)
    test30 = DataSet(test_images30, test_labels30, dtype=dtype, reshape=reshape)
    test31 = DataSet(test_images31, test_labels31, dtype=dtype, reshape=reshape)
    test32 = DataSet(test_images32, test_labels32, dtype=dtype, reshape=reshape)
    test33 = DataSet(test_images33, test_labels33, dtype=dtype, reshape=reshape)
    test34 = DataSet(test_images34, test_labels34, dtype=dtype, reshape=reshape)
    test35 = DataSet(test_images35, test_labels35, dtype=dtype, reshape=reshape)
    test36 = DataSet(test_images36, test_labels36, dtype=dtype, reshape=reshape)

    end_time_for_reading_data = datetime.now()

    time_cost_for_reading_data = end_time_for_reading_data - start_time_train_data_reading

    print("")
    print("")
    print("****************************************************************")
    print("------")
    print("time cost for reading data : " + str(time_cost_for_reading_data))
    print("------")
    print("****************************************************************")

    return Datasets_test(test=test, test0=test0, test1=test1,
                         test2=test2, test24=test24, test25=test25, test26=test26, test27=test27,
                         test28=test28, test29=test29, test30=test30, test31=test31, test32=test32,
                         test33=test33, test34=test34, test35=test35, test36=test36)
