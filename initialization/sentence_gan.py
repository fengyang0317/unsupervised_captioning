"""Train sentence gan model.

python initialization/sentence_gan.py --batch_size 512
  --save_checkpoint_steps 2000 --gen_lr 0.0001 --dis_lr 0.0001
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.gan as tfgan
import tensorflow.contrib.slim as slim
from tensorflow.contrib.gan.python.losses.python.losses_impl import modified_discriminator_loss
from tensorflow.contrib.gan.python.train import get_sequential_train_hooks

import config
from input_pipeline import AUTOTUNE
from input_pipeline import parse_sentence
from misc_fn import crop_sentence
from misc_fn import get_len
from misc_fn import transform_grads_fn
from misc_fn import validate_batch_size_for_multi_gpu
from misc_fn import variable_summaries

tf.logging.set_verbosity(tf.logging.INFO)

tf.flags.DEFINE_integer('intra_op_parallelism_threads', 0, 'Number of threads')

tf.flags.DEFINE_integer('inter_op_parallelism_threads', 0, 'Number of threads')

tf.flags.DEFINE_bool('multi_gpu', False, 'use multi gpus')

tf.flags.DEFINE_integer('emb_dim', 512, 'emb dim')

tf.flags.DEFINE_integer('mem_dim', 512, 'mem dim')

tf.flags.DEFINE_float('keep_prob', 0.8, 'keep prob')

tf.flags.DEFINE_string('job_dir', 'sen_gan', 'job dir')

tf.flags.DEFINE_integer('batch_size', 512, 'batch size')

tf.flags.DEFINE_integer('max_steps', 1000000, 'training steps')

tf.flags.DEFINE_float('weight_decay', 0, 'weight decay')

tf.flags.DEFINE_float('gen_lr', 0.0001, 'learning rate')

tf.flags.DEFINE_float('dis_lr', 0.0001, 'learning rate')

tf.flags.DEFINE_integer('save_summary_steps', 100, 'save summary steps')

tf.flags.DEFINE_integer('save_checkpoint_steps', 2000, 'save ckpt')

tf.flags.DEFINE_integer('max_caption_length', 20, 'max len')

tf.flags.DEFINE_bool('wass', False, 'use wass')

tf.flags.DEFINE_string('sae_ckpt', 'sen_ae/model.ckpt-65000', 'ckpt')

FLAGS = tf.flags.FLAGS


def generator(inputs, is_training=True):
  feat, _ = inputs
  embedding = tf.get_variable(
    name='embedding',
    shape=[FLAGS.vocab_size, FLAGS.emb_dim],
    initializer=tf.random_uniform_initializer(-0.08, 0.08))
  softmax_w = tf.matrix_transpose(embedding)
  softmax_b = tf.get_variable('softmax_b', [FLAGS.vocab_size])

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

  return sequence, logits, log_probs, baseline


def discriminator(generated_data, generator_inputs, is_training=True):
  if type(generated_data) is tuple:
    sequence = generated_data[0]
    length = get_len(sequence, FLAGS.end_id)
  else:
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
  return pred, mask


def rl_loss(gan_model, gan_loss, add_summaries):
  eps = 1e-7
  gamma = 0.9
  sequence, _, log_probs, seq_bl = gan_model.generated_data

  with tf.variable_scope(gan_model.discriminator_scope, reuse=True):
    baselines, _ = discriminator((seq_bl, None, None, None), None)
    baselines = tf.stop_gradient(baselines)

  logits, mask = gan_model.discriminator_gen_outputs

  dis_predictions = tf.nn.sigmoid(logits)
  rewards = tf.log(dis_predictions + eps)
  rewards = tf.where(mask, rewards, tf.zeros_like(rewards))

  baselines = tf.nn.sigmoid(baselines)
  baselines = tf.log(baselines + eps)
  baselines = tf.where(mask, baselines, tf.zeros_like(baselines))

  log_prob_list = tf.unstack(log_probs, axis=1)
  rewards_list = tf.unstack(rewards, axis=1)
  cumulative_rewards = []
  baseline_list = tf.unstack(baselines, axis=1)
  cumulative_baseline = []
  for t in range(FLAGS.max_caption_length):
    cum_value = tf.zeros_like(rewards_list[0])
    for s in range(t, FLAGS.max_caption_length):
      cum_value += np.power(gamma, s - t) * rewards_list[s]
    cumulative_rewards.append(cum_value)

    cum_value = tf.zeros_like(baseline_list[0])
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
    tf.summary.scalar('gen_obj', final_gen_objective)

    with tf.name_scope('rewards'):
      variable_summaries(c_rewards, mask, 'rewards')

    with tf.name_scope('advantages'):
      variable_summaries(advantages, mask, 'advantages')

    with tf.name_scope('baselines'):
      variable_summaries(c_baseline, mask, 'baselines')

    with tf.name_scope('log_probs'):
      variable_summaries(log_probs, mask, 'log_probs')

    with tf.name_scope('d_rewards'):
      variable_summaries(rewards, mask, 'd_rewards')

  return gan_loss._replace(generator_loss=final_gen_objective)


def model_fn(features, labels, mode, params):
  is_chief = not tf.get_variable_scope().reuse

  batch_size = tf.shape(labels)[0]
  noise = tf.random_normal([batch_size, FLAGS.emb_dim])
  noise = tf.nn.l2_normalize(noise, axis=1)
  gan_model = tfgan.gan_model(
    generator_fn=generator,
    discriminator_fn=discriminator,
    real_data=features[:, 1:],
    generator_inputs=(noise, labels - 1),
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
    gan_loss = tfgan.gan_loss(
      gan_model,
      generator_loss_fn=gen_loss_fn,
      discriminator_loss_fn=dis_loss_fn,
      gradient_penalty_weight=10 if FLAGS.wass else 0,
      add_summaries=is_chief)
    if is_chief:
      tfgan.eval.add_regularization_loss_summaries(gan_model)
  gan_loss = rl_loss(gan_model, gan_loss, add_summaries=is_chief)
  loss = gan_loss.generator_loss + gan_loss.discriminator_loss

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
      check_for_unused_update_ops=True,
      aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    train_op = train_ops.global_step_inc_op
    train_hooks = get_sequential_train_hooks()(train_ops)

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

  gen_var = tf.trainable_variables('Generator')
  dis_var = []
  dis_var.extend(tf.trainable_variables('Discriminator/rnn'))
  dis_var.extend(tf.trainable_variables('Discriminator/embedding'))
  saver = tf.train.Saver(gen_var + dis_var)

  def init_fn(scaffold, session):
    saver.restore(session, FLAGS.sae_ckpt)
    pass

  scaffold = tf.train.Scaffold(init_fn=init_fn)

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op,
    scaffold=scaffold,
    training_hooks=train_hooks)


def batching_func(x, batch_size):
  return x.padded_batch(
    batch_size,
    padded_shapes=(
      tf.TensorShape([None]),
      tf.TensorShape([])),
    drop_remainder=True)


def take(key, lk, sentence, ls):
  return sentence, ls


def input_fn(batch_size):
  sentence_ds = tf.data.TFRecordDataset('data/sentence.tfrec')
  sentence_ds = sentence_ds.map(parse_sentence, num_parallel_calls=AUTOTUNE)
  sentence_ds = sentence_ds.map(take, num_parallel_calls=AUTOTUNE)
  sentence_ds = sentence_ds.apply(tf.contrib.data.shuffle_and_repeat(65536))
  sentence_ds = batching_func(sentence_ds, batch_size)
  sentence_ds = sentence_ds.prefetch(AUTOTUNE)
  iterator = sentence_ds.make_one_shot_iterator()
  sentence, ls = iterator.get_next()
  return sentence, ls


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

  estimator.train(input_fn=train_input_fn, max_steps=FLAGS.max_steps)


if __name__ == '__main__':
  tf.app.run()
