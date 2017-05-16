# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# tensorboard --logdir=/Users/Pharrell_WANG/PycharmProjects/resnet_for_vcmd/resnet_model

"""ResNet Train/Eval module.
"""
import time
import six
import sys

import numpy as np
import md_resnet_model
import tensorflow as tf
from md_input import read_train_data_sets, read_test_data_sets

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('mode', 'eval', 'train or eval.')
tf.app.flags.DEFINE_string('train_data_path',
                           '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/train_data_32x32/training_32x32_equal.csv',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path',
                           '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32.csv',
                           'Filepattern for eval data')
# tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', '/Users/Pharrell_WANG/PycharmProjects/resnet_for_vcmd/32x32_wrn_model/train',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '/Users/Pharrell_WANG/PycharmProjects/resnet_for_vcmd/32x32_wrn_model/eval',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 1000,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', True,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', '/Users/Pharrell_WANG/PycharmProjects/resnet_for_vcmd/32x32_wrn_model',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')

image_width = 32

CLASSES = 37


def evaluate(hps):
    """Eval loop."""
    md = read_test_data_sets()
    with tf.name_scope('eval_input'):
        images = tf.placeholder(tf.float32, [100, image_width, image_width, 1], name='eval-batch-images-input')
        # correct answers go here
        labels = tf.placeholder(tf.float32, [100, CLASSES], name='eval-correct-labels-input')

    # images = md.test.images
    # labels = md.test.labels
    model = md_resnet_model.ResNet(hps, images, labels, 'eval')
    model.build_graph()

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # tf.train.start_queue_runners(sess)

    best_precision = 0.0
    while True:
        try:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)
            continue
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
            continue
        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        total_prediction, correct_prediction, precision_x = 0, 0, 0
        # cnt = 0
        for _ in six.moves.range(FLAGS.eval_batch_count):
            # cnt += 1
            # print('new one---->     ' + str(cnt))
            batch_X, batch_Y = md.test.next_batch(100)
            (summaries, loss, predictions, truth, train_step) = sess.run(
                [model.summaries, model.cost, model.predictions,
                 model.labels, model.global_step], {images: batch_X, labels: batch_Y})

            truth = np.argmax(truth, axis=1)

            # print(truth)

            predictions = np.argmax(predictions, axis=1)

            # print(predictions)
            # print(truth == predictions)
            # print(predictions.shape[0])

            precision_x = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

            print(sess.run(precision_x))

            correct_prediction += np.sum(truth == predictions)
            total_prediction += predictions.shape[0]
            # print('----')
            # print(correct_prediction)
            # print(total_prediction)

        precision = 1.0 * correct_prediction / total_prediction
        best_precision = max(precision, best_precision)

        precision_summ = tf.Summary()
        precision_summ.value.add(
            tag='Precision', simple_value=precision)
        summary_writer.add_summary(precision_summ, train_step)
        best_precision_summ = tf.Summary()
        best_precision_summ.value.add(
            tag='Best Precision', simple_value=best_precision)
        summary_writer.add_summary(best_precision_summ, train_step)
        summary_writer.add_summary(summaries, train_step)
        tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f' %
                        (loss, precision, best_precision))
        summary_writer.flush()

        if FLAGS.eval_once:
            break

        time.sleep(60)


def main(_):
    dev = '/gpu:0'

    batch_size = 100

    num_classes = 37

    hps = md_resnet_model.HParams(batch_size=batch_size,
                                  num_classes=num_classes,
                                  min_lrn_rate=0.0001,
                                  lrn_rate=0.1,
                                  # num_residual_units=5,
                                  num_residual_units=4,
                                  use_bottleneck=False,
                                  weight_decay_rate=0.0002,
                                  relu_leakiness=0.1,
                                  optimizer='mom')

    with tf.device(dev):
        evaluate(hps)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
