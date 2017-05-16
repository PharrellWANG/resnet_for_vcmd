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

"""ResNet Train/Eval module.
"""
import time
import six
import sys

import cifar_input
import numpy as np
import resnet_model
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('mode', 'eval', 'train or eval.')
tf.app.flags.DEFINE_string('train_data_path', '/Users/Pharrell_WANG/RESNET/cifar10/data_batch*',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', '/Users/Pharrell_WANG/RESNET/cifar10/test_batch.bin',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', '/Users/Pharrell_WANG/PycharmProjects/resnet_for_vcmd/original_resnet_model/train',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '/Users/Pharrell_WANG/PycharmProjects/resnet_for_vcmd/original_resnet_model/eval',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 50,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', '/Users/Pharrell_WANG/PycharmProjects/resnet_for_vcmd/original_resnet_model',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 0,
                            'Number of gpus used for training. (0 or 1)')


images, labels = cifar_input.build_input(
        FLAGS.dataset, FLAGS.eval_data_path, 10, FLAGS.mode)
print('-----======-----')
print(images.shape)
print(labels.shape)

print(images[3:5])


# def main(_):
#     if FLAGS.num_gpus == 0:
#         dev = '/cpu:0'
#     elif FLAGS.num_gpus == 1:
#         dev = '/gpu:0'
#     else:
#         raise ValueError('Only support 0 or 1 gpu.')
#
#     if FLAGS.mode == 'train':
#         batch_size = 128
#     elif FLAGS.mode == 'eval':
#         batch_size = 100
#
#     if FLAGS.dataset == 'cifar10':
#         num_classes = 10
#     elif FLAGS.dataset == 'cifar100':
#         num_classes = 100
#
#     hps = resnet_model.HParams(batch_size=batch_size,
#                                num_classes=num_classes,
#                                min_lrn_rate=0.0001,
#                                lrn_rate=0.1,
#                                num_residual_units=5,
#                                use_bottleneck=False,
#                                weight_decay_rate=0.0002,
#                                relu_leakiness=0.1,
#                                optimizer='mom')
#
#     with tf.device(dev):
#         if FLAGS.mode == 'train':
#             train(hps)
#         elif FLAGS.mode == 'eval':
#             evaluate(hps)


# if __name__ == '__main__':
#     tf.logging.set_verbosity(tf.logging.INFO)
#     tf.app.run()
