"""Generate pseudo captions.

python gen_obj2sen_caption.py --num_proc 64
"""

import multiprocessing
import os
from functools import partial

from absl import app
from absl import flags

from misc_fn import _int64_feature_list

flags.DEFINE_integer('num_proc', 1, 'number of processes')

flags.DEFINE_integer('num_gpus', 1, 'number of gpus')

from sentence_infer import Infer

FLAGS = flags.FLAGS


def initializer():
  if FLAGS.num_gpus > 0:
    current = multiprocessing.current_process()
    id = current._identity[0] - 1
    os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % (id % FLAGS.num_gpus)
  else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
  global infer
  infer = Infer()


def run(classes):
  tf = infer.tf
  sentences = infer.infer(classes[::-1])
  sentence = sentences[0][0].split()
  sentence = [infer.vocab.word_to_id(i) for i in sentence]
  context = tf.train.Features()
  feature_lists = tf.train.FeatureLists(feature_list={
    'sentence': _int64_feature_list(sentence)
  })
  sequence_example = tf.train.SequenceExample(
    context=context, feature_lists=feature_lists)
  return sequence_example.SerializeToString()


def parse_image(serialized, tf):
  """Parses a tensorflow.SequenceExample into an image and detected objects.

  Args:
    serialized: A scalar string Tensor; a single serialized SequenceExample.

  Returns:
    encoded_image: A scalar string Tensor containing a JPEG encoded image.
    classes: A 1-D int64 Tensor containing the detected objects.
    scores: A 1-D float32 Tensor containing the detection scores.
  """
  context, sequence = tf.parse_single_sequence_example(
    serialized,
    sequence_features={
      'classes': tf.FixedLenSequenceFeature([], dtype=tf.int64),
      'scores': tf.FixedLenSequenceFeature([], dtype=tf.float32),
    })

  classes = tf.to_int32(sequence['classes'])
  scores = sequence['scores']
  return classes, scores


def image_generator(tf):
  ds = tf.data.TFRecordDataset('data/image_train.tfrec')
  ds = ds.map(
    partial(parse_image, tf=tf),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
  for classes, scores in ds:
    yield classes.numpy()


def main(_):
  pool = multiprocessing.Pool(FLAGS.num_proc, initializer=initializer)
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  import tensorflow as tf
  tf.enable_eager_execution()
  with tf.python_io.TFRecordWriter('data/obj2sen_captions.tfrec') as writer:
    for i in pool.imap(run, image_generator(tf)):
      writer.write(i)


if __name__ == '__main__':
  app.run(main)
