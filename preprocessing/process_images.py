"""Convert the images and the detected object labels to tfrecords."""
import os
import pickle as pkl
import random

import h5py
import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from functools import partial

from misc_fn import _bytes_feature
from misc_fn import _float_feature_list
from misc_fn import _int64_feature_list

tf.enable_eager_execution()

flags.DEFINE_string('image_path', None, 'Path to all coco images.')

FLAGS = flags.FLAGS


def image_generator(split):
  with open('data/coco_%s.txt' % split, 'r') as f:
    filename = list(f)
    filename = [i.strip() for i in filename]
  if split == 'train':
    random.shuffle(filename)
  with open('data/all_ids.pkl', 'r') as f:
    all_ids = pkl.load(f)
  with h5py.File('data/object.hdf5', 'r') as f:
    for i in filename:
      name = os.path.splitext(i)[0]
      detection_classes = f[name + '/detection_classes'][:].astype(np.int32)
      detection_scores = f[name + '/detection_scores'][:]
      detection_classes, ind = np.unique(detection_classes, return_index=True)
      detection_scores = detection_scores[ind]
      detection_classes = [all_ids[j] for j in detection_classes]
      image_path = FLAGS.image_path + '/' + i
      with tf.gfile.FastGFile(image_path, 'r') as g:
        image = g.read()
      context = tf.train.Features(feature={
        'image/name': _bytes_feature(i),
        'image/data': _bytes_feature(image),
      })
      feature_lists = tf.train.FeatureLists(feature_list={
        'classes': _int64_feature_list(detection_classes),
        'scores': _float_feature_list(detection_scores)
      })
      sequence_example = tf.train.SequenceExample(
        context=context, feature_lists=feature_lists)

      yield sequence_example.SerializeToString()


def gen_tfrec(split):
  ds = tf.data.Dataset.from_generator(partial(image_generator, split=split),
                                      output_types=tf.string, output_shapes=())
  tfrec = tf.data.experimental.TFRecordWriter('data/image_%s.tfrec' % split)
  tfrec.write(ds)


def main(_):
  for i in ['train', 'val', 'test']:
    gen_tfrec(i)


if __name__ == '__main__':
  app.run(main)
