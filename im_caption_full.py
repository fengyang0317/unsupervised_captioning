"""Train the full model.

python im_caption_full.py --multi_gpu --batch_size 512 --save_checkpoint_steps\
  1000 --gen_lr 0.001 --dis_lr 0.001
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow.contrib.gan as tfgan
import tensorflow.contrib.slim as slim
from tensorflow.contrib.framework import nest
from tensorflow.contrib.gan.python.losses.python.losses_impl import modified_discriminator_loss
from tensorflow.contrib.gan.python.train import get_sequential_train_hooks

from config import TF_MODELS_PATH
from input_pipeline import input_fn
from misc_fn import crop_sentence
from misc_fn import get_len
from misc_fn import obj_rewards
from misc_fn import transform_grads_fn
from misc_fn import validate_batch_size_for_multi_gpu
from misc_fn import variable_summaries

sys.path.append(TF_MODELS_PATH + '/research/slim')
from nets import inception_v4

tf.logging.set_verbosity(tf.logging.INFO)

tf.flags.DEFINE_integer('intra_op_parallelism_threads', 0, 'Number of threads')

tf.flags.DEFINE_integer('inter_op_parallelism_threads', 0, 'Number of threads')

tf.flags.DEFINE_bool('multi_gpu', False, 'use multi gpus')

tf.flags.DEFINE_integer('emb_dim', 512, 'emb dim')

tf.flags.DEFINE_integer('mem_dim', 512, 'mem dim')

tf.flags.DEFINE_float('keep_prob', 0.8, 'keep prob')

tf.flags.DEFINE_string('job_dir', 'saving', 'job dir')

tf.flags.DEFINE_integer('batch_size', 64, 'batch size')

tf.flags.DEFINE_integer('max_steps', 1000000, 'maximum training steps')

tf.flags.DEFINE_float('gen_lr', 0.0001, 'learning rate')

tf.flags.DEFINE_float('dis_lr', 0.0001, 'learning rate')

tf.flags.DEFINE_integer('save_summary_steps', 100, 'save summary steps')

tf.flags.DEFINE_integer('save_checkpoint_steps', 5000, 'save ckpt')

tf.flags.DEFINE_integer('max_caption_length', 20, 'max len')

tf.flags.DEFINE_bool('wass', False, 'use wass')

tf.flags.DEFINE_bool('use_pool', False, 'use pool')

tf.flags.DEFINE_integer('pool_size', 512, 'pool size')

tf.flags.DEFINE_string('inc_ckpt', None, 'path to InceptionV4 checkpoint')

tf.flags.DEFINE_string('imcap_ckpt', None, 'initialization checkpoint')

tf.flags.DEFINE_string('sae_ckpt', None, 'initialization checkpoint')

tf.flags.DEFINE_float('w_obj', 10, 'object weight')

tf.flags.DEFINE_float('w_mse', 100, 'object weight')

FLAGS = tf.flags.FLAGS


def generator(inputs, is_training=True):
  """The sentence generator."""
  embedding = tf.get_variable(
    name='embedding',
    shape=[FLAGS.vocab_size, FLAGS.emb_dim],
    initializer=tf.random_uniform_initializer(-0.08, 0.08))
  softmax_w = tf.matrix_transpose(embedding)
  softmax_b = tf.get_variable('softmax_b', [FLAGS.vocab_size])

  inputs = inputs[0]
  feat = slim.fully_connected(inputs, FLAGS.mem_dim, activation_fn=None)
  feat = tf.nn.l2_normalize(feat, axis=1)

  batch_size = tf.shape(feat)[0]
  cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.mem_dim)
  if is_training:
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, FLAGS.keep_prob, FLAGS.keep_prob)
  zero_state = cell.zero_state(batch_size, tf.float32)

  sequence, logits, log_probs, rnn_outs = [], [], [], []

  _, state = cell(feat, zero_state)
  state_bl = state
  tf.get_variable_scope().reuse_variables()
  for t in range(FLAGS.max_caption_length):
    if t == 0:
      rnn_inp = tf.zeros([batch_size], tf.int32) + FLAGS.start_id
    rnn_inp = tf.nn.embedding_lookup(embedding, rnn_inp)
    rnn_out, state = cell(rnn_inp, state)
    rnn_outs.append(rnn_out)
    logit = tf.nn.bias_add(tf.matmul(rnn_out, softmax_w), softmax_b)
    categorical = tf.contrib.distributions.Categorical(logits=logit)
    fake = categorical.sample()
    log_prob = categorical.log_prob(fake)
    sequence.append(fake)
    log_probs.append(log_prob)
    logits.append(logit)
    rnn_inp = fake
  sequence = tf.stack(sequence, axis=1)
  log_probs = tf.stack(log_probs, axis=1)
  logits = tf.stack(logits, axis=1)

  # Computes the baseline for self-critic.
  baseline = []
  state = state_bl
  for t in range(FLAGS.max_caption_length):
    if t == 0:
      rnn_inp = tf.zeros([batch_size], tf.int32) + FLAGS.start_id
    rnn_inp = tf.nn.embedding_lookup(embedding, rnn_inp)
    rnn_out, state = cell(rnn_inp, state)
    logit = tf.nn.bias_add(tf.matmul(rnn_out, softmax_w), softmax_b)
    fake = tf.argmax(logit, axis=1, output_type=tf.int32)
    baseline.append(fake)
    rnn_inp = fake
  baseline = tf.stack(baseline, axis=1)

  return sequence, logits, log_probs, baseline, feat


def discriminator(generated_data, generator_inputs, is_training=True):
  """The discriminator."""
  if type(generated_data) is tuple:
    # When the sentences are generated, we need to compute their length.
    sequence = generated_data[0]
    length = get_len(sequence, FLAGS.end_id)
  else:
    # We already know the length of the sentences from the input pipeline.
    sequence = generated_data
    length = generator_inputs[-1]
  embedding = tf.get_variable(
    name='embedding',
    shape=[FLAGS.vocab_size, FLAGS.emb_dim],
    initializer=tf.random_uniform_initializer(-0.08, 0.08))
  cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.mem_dim)
  if is_training:
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, FLAGS.keep_prob, FLAGS.keep_prob)

  rnn_inputs = tf.nn.embedding_lookup(embedding, sequence)
  rnn_out, state = tf.nn.dynamic_rnn(cell, rnn_inputs, length, dtype=tf.float32)
  pred = slim.fully_connected(rnn_out, 1, activation_fn=None, scope='fc')
  pred = tf.squeeze(pred, 2)
  mask = tf.sequence_mask(length, tf.shape(sequence)[1])

  idx = tf.transpose(tf.stack([tf.range(tf.shape(length)[0]), length - 1]))
  state_h = tf.gather_nd(rnn_out, idx)
  feat = slim.fully_connected(state_h, FLAGS.mem_dim, activation_fn=None,
                              scope='recon')
  feat = tf.nn.l2_normalize(feat, axis=1)
  return pred, mask, feat


def rl_loss(gan_model, gan_loss, classes, scores, num, add_summaries):
  """Reinforcement learning loss."""
  eps = 1e-7
  gamma = 0.9
  sequence, _, log_probs, seq_bl, pca = gan_model.generated_data

  with tf.variable_scope(gan_model.discriminator_scope, reuse=True):
    baselines, _, feat_bl = discriminator((seq_bl, None, None, None, pca), None)
    baselines, feat_bl = nest.map_structure(
      tf.stop_gradient, (baselines, feat_bl))

  logits, mask, feat = gan_model.discriminator_gen_outputs

  dist = tf.reduce_mean(tf.squared_difference(pca, feat), axis=1,
                        keepdims=True) * FLAGS.w_mse
  loss_mse = tf.reduce_mean(dist)
  l_rewards = -dist
  l_rewards = tf.tile(l_rewards, [1, sequence.shape[1]])
  l_rewards = tf.where(mask, l_rewards, tf.zeros_like(l_rewards))
  l_rewards_mat = l_rewards
  l_rewards = tf.unstack(l_rewards, axis=1)

  dis_predictions = tf.nn.sigmoid(logits)
  d_rewards = tf.log(dis_predictions + eps)
  o_rewards = obj_rewards(sequence, mask, classes, scores, num) * FLAGS.w_obj
  rewards = d_rewards + o_rewards
  rewards = tf.where(mask, rewards, tf.zeros_like(rewards))

  l_bl = -tf.reduce_mean(tf.squared_difference(pca, feat_bl), axis=1,
                         keepdims=True) * FLAGS.w_mse
  l_bl = tf.tile(l_bl, [1, seq_bl.shape[1]])
  l_bl = tf.where(mask, l_bl, tf.zeros_like(l_bl))
  l_bl = tf.unstack(l_bl, axis=1)
  baselines = tf.nn.sigmoid(baselines)
  baselines = tf.log(baselines + eps)
  baselines += obj_rewards(seq_bl, mask, classes, scores, num) * FLAGS.w_obj
  baselines = tf.where(mask, baselines, tf.zeros_like(baselines))

  log_prob_list = tf.unstack(log_probs, axis=1)
  rewards_list = tf.unstack(rewards, axis=1)
  cumulative_rewards = []
  baseline_list = tf.unstack(baselines, axis=1)
  cumulative_baseline = []
  for t in range(FLAGS.max_caption_length):
    cum_value = l_rewards[t]
    for s in range(t, FLAGS.max_caption_length):
      cum_value += np.power(gamma, s - t) * rewards_list[s]
    cumulative_rewards.append(cum_value)

    cum_value = l_bl[t]
    for s in range(t, FLAGS.max_caption_length):
      cum_value += np.power(gamma, s - t) * baseline_list[s]
    cumulative_baseline.append(cum_value)
  c_rewards = tf.stack(cumulative_rewards, axis=1)
  c_baseline = tf.stack(cumulative_baseline, axis=1)

  advantages = []
  final_gen_objective = []
  for t in range(FLAGS.max_caption_length):
    log_probability = log_prob_list[t]
    cum_advantage = cumulative_rewards[t] - cumulative_baseline[t]
    cum_advantage = tf.clip_by_value(cum_advantage, -5.0, 5.0)
    advantages.append(cum_advantage)
    final_gen_objective.append(
      log_probability * tf.stop_gradient(cum_advantage))
  final_gen_objective = tf.stack(final_gen_objective, axis=1)
  final_gen_objective = tf.losses.compute_weighted_loss(final_gen_objective,
                                                        tf.to_float(mask))
  final_gen_objective = -final_gen_objective
  advantages = tf.stack(advantages, axis=1)

  if add_summaries:
    tf.summary.scalar('losses/mse', loss_mse)
    tf.summary.scalar('losses/gen_obj', final_gen_objective)
    with tf.name_scope('rewards'):
      variable_summaries(c_rewards, mask, 'rewards')

    with tf.name_scope('advantages'):
      variable_summaries(advantages, mask, 'advantages')

    with tf.name_scope('baselines'):
      variable_summaries(c_baseline, mask, 'baselines')

    with tf.name_scope('log_probs'):
      variable_summaries(log_probs, mask, 'log_probs')

    with tf.name_scope('d_rewards'):
      variable_summaries(d_rewards, mask, 'd_rewards')

    with tf.name_scope('l_rewards'):
      variable_summaries(l_rewards_mat, mask, 'l_rewards')

    with tf.name_scope('o_rewards'):
      variable_summaries(o_rewards, mask, 'o_rewards')
      o_rewards = tf.where(mask, o_rewards, tf.zeros_like(o_rewards))
      minimum = tf.minimum(tf.reduce_min(o_rewards, axis=1, keepdims=True), 0.0)
      o_rewards = tf.reduce_sum(
        tf.to_float(tf.logical_and(o_rewards > minimum, mask)), axis=1)
      o_rewards = tf.reduce_mean(o_rewards)
      tf.summary.scalar('mean_found_obj', o_rewards)

  return gan_loss._replace(generator_loss=final_gen_objective,
                           discriminator_loss=gan_loss.discriminator_loss + loss_mse)


def sentence_ae(gan_model, features, labels, add_summaries=True):
  """Sentence auto-encoder."""
  with tf.variable_scope(gan_model.discriminator_scope, reuse=True):
    feat = discriminator(features['key'], [None, features['lk']])[2]
  with tf.variable_scope(gan_model.generator_scope, reuse=True):
    embedding = tf.get_variable(
      name='embedding',
      shape=[FLAGS.vocab_size, FLAGS.emb_dim],
      initializer=tf.random_uniform_initializer(-0.08, 0.08))
    softmax_w = tf.matrix_transpose(embedding)
    softmax_b = tf.get_variable('softmax_b', [FLAGS.vocab_size])

    sentence, ls = labels['sentence'], labels['len']
    targets = sentence[:, 1:]
    sentence = sentence[:, :-1]
    ls -= 1
    sentence = tf.nn.embedding_lookup(embedding, sentence)

    batch_size = tf.shape(feat)[0]
    cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.mem_dim)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, FLAGS.keep_prob, FLAGS.keep_prob)
    zero_state = cell.zero_state(batch_size, tf.float32)
    _, state = cell(feat, zero_state)
    tf.get_variable_scope().reuse_variables()
    out, state = tf.nn.dynamic_rnn(cell, sentence, ls, state)
    out = tf.reshape(out, [-1, FLAGS.mem_dim])
    logits = tf.nn.bias_add(tf.matmul(out, softmax_w), softmax_b)
    logits = tf.reshape(logits, [batch_size, -1, FLAGS.vocab_size])

  mask = tf.sequence_mask(ls, tf.shape(sentence)[1])
  targets = tf.boolean_mask(targets, mask)
  logits = tf.boolean_mask(logits, mask)
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                        logits=logits)
  loss = tf.reduce_mean(loss)
  if add_summaries:
    tf.summary.scalar('losses/sentence_ae', loss)
  return loss


def model_fn(features, labels, mode, params):
  """The full unsupervised captioning model."""
  is_chief = not tf.get_variable_scope().reuse

  with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
    net, _ = inception_v4.inception_v4(features['im'], None, is_training=False)
  net = tf.squeeze(net, [1, 2])
  inc_saver = tf.train.Saver(tf.global_variables('InceptionV4'))

  gan_model = tfgan.gan_model(
    generator_fn=generator,
    discriminator_fn=discriminator,
    real_data=labels['sentence'][:, 1:],
    generator_inputs=(net, labels['len'] - 1),
    check_shapes=False)

  if is_chief:
    for variable in tf.trainable_variables():
      tf.summary.histogram(variable.op.name, variable)
    tf.summary.histogram('logits/gen_logits',
                         gan_model.discriminator_gen_outputs[0])
    tf.summary.histogram('logits/real_logits',
                         gan_model.discriminator_real_outputs[0])

  def gen_loss_fn(gan_model, add_summaries):
    return 0

  def dis_loss_fn(gan_model, add_summaries):
    discriminator_real_outputs = gan_model.discriminator_real_outputs
    discriminator_gen_outputs = gan_model.discriminator_gen_outputs
    real_logits = tf.boolean_mask(discriminator_real_outputs[0],
                                  discriminator_real_outputs[1])
    gen_logits = tf.boolean_mask(discriminator_gen_outputs[0],
                                 discriminator_gen_outputs[1])
    return modified_discriminator_loss(real_logits, gen_logits,
                                       add_summaries=add_summaries)

  with tf.name_scope('losses'):
    pool_fn = functools.partial(tfgan.features.tensor_pool,
                                pool_size=FLAGS.pool_size)
    gan_loss = tfgan.gan_loss(
      gan_model,
      generator_loss_fn=gen_loss_fn,
      discriminator_loss_fn=dis_loss_fn,
      gradient_penalty_weight=10 if FLAGS.wass else 0,
      tensor_pool_fn=pool_fn if FLAGS.use_pool else None,
      add_summaries=is_chief)
    if is_chief:
      tfgan.eval.add_regularization_loss_summaries(gan_model)
  gan_loss = rl_loss(gan_model, gan_loss, features['classes'],
                     features['scores'], features['num'],
                     add_summaries=is_chief)
  sen_ae_loss = sentence_ae(gan_model, features, labels, is_chief)
  loss = gan_loss.generator_loss + gan_loss.discriminator_loss + sen_ae_loss
  gan_loss = gan_loss._replace(
    generator_loss=gan_loss.generator_loss + sen_ae_loss)

  with tf.name_scope('train'):
    gen_opt = tf.train.AdamOptimizer(params.gen_lr, 0.5)
    dis_opt = tf.train.AdamOptimizer(params.dis_lr, 0.5)
    if params.multi_gpu:
      gen_opt = tf.contrib.estimator.TowerOptimizer(gen_opt)
      dis_opt = tf.contrib.estimator.TowerOptimizer(dis_opt)
    train_ops = tfgan.gan_train_ops(
      gan_model,
      gan_loss,
      generator_optimizer=gen_opt,
      discriminator_optimizer=dis_opt,
      transform_grads_fn=transform_grads_fn,
      summarize_gradients=is_chief,
      check_for_unused_update_ops=not FLAGS.use_pool,
      aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    train_op = train_ops.global_step_inc_op
    train_hooks = get_sequential_train_hooks()(train_ops)

  # Summary the generated caption on the fly.
  if is_chief:
    with open('data/word_counts.txt', 'r') as f:
      dic = list(f)
      dic = [i.split()[0] for i in dic]
      dic.append('<unk>')
      dic = tf.convert_to_tensor(dic)
    sentence = crop_sentence(gan_model.generated_data[0][0], FLAGS.end_id)
    sentence = tf.gather(dic, sentence)
    real = crop_sentence(gan_model.real_data[0], FLAGS.end_id)
    real = tf.gather(dic, real)
    train_hooks.append(
      tf.train.LoggingTensorHook({'fake': sentence, 'real': real},
                                 every_n_iter=100))
    tf.summary.text('fake', sentence)
    tf.summary.image('im', features['im'][None, 0])

  gen_saver = tf.train.Saver(tf.trainable_variables('Generator'))
  dis_var = []
  dis_var.extend(tf.trainable_variables('Discriminator/rnn'))
  dis_var.extend(tf.trainable_variables('Discriminator/embedding'))
  dis_var.extend(tf.trainable_variables('Discriminator/fc'))
  dis_saver = tf.train.Saver(dis_var)

  def init_fn(scaffold, session):
    inc_saver.restore(session, FLAGS.inc_ckpt)
    if FLAGS.imcap_ckpt:
      gen_saver.restore(session, FLAGS.imcap_ckpt)
    if FLAGS.sae_ckpt:
      dis_saver.restore(session, FLAGS.sae_ckpt)

  scaffold = tf.train.Scaffold(init_fn=init_fn)

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op,
    scaffold=scaffold,
    training_hooks=train_hooks)


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
