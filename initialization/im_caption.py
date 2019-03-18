"""Train sentence gan model.

python im_caption.py --batch_size 512 --multi_gpu --save_checkpoint_steps 2000
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import sys

import tensorflow as tf
import tensorflow.contrib.slim as slim

from config import TF_MODELS_PATH
from misc_fn import crop_sentence
from misc_fn import transform_grads_fn
from misc_fn import validate_batch_size_for_multi_gpu
from input_pipeline import parse_image
from input_pipeline import preprocess_image
from input_pipeline import AUTOTUNE
from input_pipeline import parse_sentence

sys.path.append(TF_MODELS_PATH + '/research/slim')
from nets import inception_v4

tf.logging.set_verbosity(tf.logging.INFO)

tf.flags.DEFINE_integer('intra_op_parallelism_threads', 0, 'Number of threads')

tf.flags.DEFINE_integer('inter_op_parallelism_threads', 0, 'Number of threads')

tf.flags.DEFINE_bool('multi_gpu', False, 'use multi gpus')

tf.flags.DEFINE_integer('emb_dim', 512, 'emb dim')

tf.flags.DEFINE_integer('mem_dim', 512, 'mem dim')

tf.flags.DEFINE_float('keep_prob', 0.8, 'keep prob')

tf.flags.DEFINE_string('job_dir', 'saving_imcap', 'job dir')

tf.flags.DEFINE_integer('batch_size', 512, 'batch size')

tf.flags.DEFINE_integer('max_steps', 1000000, 'training steps')

tf.flags.DEFINE_float('weight_decay', 0, 'weight decay')

tf.flags.DEFINE_float('lr', 0.001, 'learning rate')

tf.flags.DEFINE_integer('save_summary_steps', 100, 'save summary steps')

tf.flags.DEFINE_integer('save_checkpoint_steps', 2000, 'save ckpt')

tf.flags.DEFINE_string('inc_ckpt', None, 'InceptionV4 checkpoint')

tf.flags.DEFINE_string('o2s_ckpt', None, 'ckpt')

FLAGS = tf.flags.FLAGS


def model_fn(features, labels, mode, params):
  is_chief = not tf.get_variable_scope().reuse
  is_training = mode == tf.estimator.ModeKeys.TRAIN
  batch_size = tf.shape(features)[0]

  with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
    net, _ = inception_v4.inception_v4(features, None, is_training=False)
  net = tf.squeeze(net, [1, 2])
  inc_saver = tf.train.Saver(tf.global_variables('InceptionV4'))

  with tf.variable_scope('Generator'):
    feat = slim.fully_connected(net, FLAGS.mem_dim, activation_fn=None)
    feat = tf.nn.l2_normalize(feat, axis=1)
    sentence, ls = labels['sentence'], labels['len']
    targets = sentence[:, 1:]
    sentence = sentence[:, :-1]
    ls -= 1

    embedding = tf.get_variable(
      name='embedding',
      shape=[FLAGS.vocab_size, FLAGS.emb_dim],
      initializer=tf.random_uniform_initializer(-0.08, 0.08))
    softmax_w = tf.matrix_transpose(embedding)
    softmax_b = tf.get_variable('softmax_b', [FLAGS.vocab_size])
    sentence = tf.nn.embedding_lookup(embedding, sentence)

    cell = tf.nn.rnn_cell.BasicLSTMCell(params.mem_dim)
    if is_training:
      cell = tf.nn.rnn_cell.DropoutWrapper(cell, params.keep_prob,
                                           params.keep_prob)
    zero_state = cell.zero_state(batch_size, tf.float32)
    _, state = cell(feat, zero_state)
    tf.get_variable_scope().reuse_variables()
    out, state = tf.nn.dynamic_rnn(cell, sentence, ls, state)
    out = tf.reshape(out, [-1, FLAGS.mem_dim])
    logits = tf.nn.bias_add(tf.matmul(out, softmax_w), softmax_b)
    logits = tf.reshape(logits, [batch_size, -1, FLAGS.vocab_size])
    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)

  mask = tf.sequence_mask(ls, tf.shape(sentence)[1])
  targets = tf.boolean_mask(targets, mask)
  logits = tf.boolean_mask(logits, mask)
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                        logits=logits)
  loss = tf.reduce_mean(loss)

  opt = tf.train.AdamOptimizer(params.lr)
  if params.multi_gpu:
    opt = tf.contrib.estimator.TowerOptimizer(opt)
  grads = opt.compute_gradients(loss, tf.trainable_variables('Generator'))
  grads[2] = (tf.convert_to_tensor(grads[2][0]), grads[2][1])
  for i in range(2, len(grads)):
    grads[i] = (grads[i][0] * 0.1, grads[i][1])
  grads = transform_grads_fn(grads)
  train_op = opt.apply_gradients(grads, global_step=tf.train.get_global_step())

  train_hooks = None
  if is_chief:
    with open('data/word_counts.txt', 'r') as f:
      dic = list(f)
      dic = [i.split()[0] for i in dic]
      end_id = dic.index('</S>')
      dic.append('<unk>')
      dic = tf.convert_to_tensor(dic)
    sentence = crop_sentence(predictions[0], end_id)
    sentence = tf.gather(dic, sentence)
    tf.summary.text('fake', sentence)
    tf.summary.image('im', features[None, 0])
    for variable in tf.trainable_variables():
      tf.summary.histogram(variable.op.name, variable)

  predictions = tf.boolean_mask(predictions, mask)
  metrics = {
    'acc': tf.metrics.accuracy(targets, predictions)
  }

  gen_var = tf.trainable_variables('Generator')[2:]
  gen_saver = tf.train.Saver(gen_var)

  def init_fn(scaffold, session):
    inc_saver.restore(session, FLAGS.inc_ckpt)
    if FLAGS.o2s_ckpt:
      gen_saver.restore(session, FLAGS.o2s_ckpt)

  scaffold = tf.train.Scaffold(init_fn=init_fn)

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op,
    scaffold=scaffold,
    training_hooks=train_hooks,
    eval_metric_ops=metrics)


def batching_func(x, batch_size):
  return x.padded_batch(
    batch_size,
    padded_shapes=(
      tf.TensorShape([299, 299, 3]),
      tf.TensorShape([None]),
      tf.TensorShape([])))


def take(image, sentence):
  sentence = tf.concat([[FLAGS.start_id], sentence[2], [FLAGS.end_id]], axis=0)
  return image[0], sentence, tf.shape(sentence)[0]


def input_fn(batch_size):
  image_ds = tf.data.TFRecordDataset('data/image_train.tfrec')
  image_ds = image_ds.map(parse_image, num_parallel_calls=AUTOTUNE)
  image_ds = image_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)

  sentence_ds = tf.data.TFRecordDataset('data/obj2sen_captions.tfrec')
  sentence_ds = sentence_ds.map(parse_sentence, num_parallel_calls=AUTOTUNE)

  dataset = tf.data.Dataset.zip((image_ds, sentence_ds))
  dataset = dataset.map(take)
  dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(4096))
  dataset = batching_func(dataset, batch_size)
  dataset = dataset.prefetch(AUTOTUNE)
  iterator = dataset.make_one_shot_iterator()
  im, sentence, ls = iterator.get_next()
  return im, {'sentence': sentence, 'len': ls}


def main(_):
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  if FLAGS.multi_gpu:
    validate_batch_size_for_multi_gpu(FLAGS.batch_size)
    model_function = tf.contrib.estimator.replicate_model_fn(
      model_fn,
      loss_reduction=tf.losses.Reduction.MEAN)
  else:
    model_function = model_fn

  sess_config = tf.ConfigProto(
    allow_soft_placement=True,
    intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads,
    inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
    gpu_options=tf.GPUOptions(allow_growth=True))

  run_config = tf.estimator.RunConfig(
    session_config=sess_config,
    save_checkpoints_steps=FLAGS.save_checkpoint_steps,
    save_summary_steps=FLAGS.save_summary_steps,
    keep_checkpoint_max=100)

  train_input_fn = functools.partial(input_fn, batch_size=FLAGS.batch_size)

  estimator = tf.estimator.Estimator(
    model_fn=model_function,
    model_dir=FLAGS.job_dir,
    config=run_config,
    params=FLAGS)

  estimator.train(train_input_fn, max_steps=FLAGS.max_steps)


if __name__ == '__main__':
  tf.app.run()
