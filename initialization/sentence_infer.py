"""Given some object words, infers a whole sentence."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys

import numpy as np
import tensorflow as tf

from config import TF_MODELS_PATH

sys.path.append(TF_MODELS_PATH + '/research/im2txt/')
from im2txt.inference_utils.caption_generator import Caption
from im2txt.inference_utils.caption_generator import TopN
from im2txt.inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('job_dir', 'obj2sen', 'job dir')

tf.flags.DEFINE_integer('emb_dim', 512, 'emb dim')

tf.flags.DEFINE_integer('mem_dim', 512, 'mem dim')

tf.flags.DEFINE_float('keep_prob', 0.8, 'keep prob')

tf.flags.DEFINE_integer('batch_size', 1, 'batch size')

tf.flags.DEFINE_string("vocab_file", "data/word_counts.txt",
                       "Text file containing the vocabulary.")

tf.flags.DEFINE_integer('beam_size', 3, 'beam size')

tf.flags.DEFINE_integer('max_caption_length', 20, 'beam size')

tf.flags.DEFINE_float('length_normalization_factor', 0.0, 'l n f')


def _tower_fn(key, lk, is_training=False):
  with tf.variable_scope('Discriminator'):
    embedding = tf.get_variable(
      name='embedding',
      shape=[FLAGS.vocab_size, FLAGS.emb_dim],
      initializer=tf.random_uniform_initializer(-0.08, 0.08))

    key = tf.nn.embedding_lookup(embedding, key)

    cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.mem_dim)
    if is_training:
      cell = tf.nn.rnn_cell.DropoutWrapper(cell, FLAGS.keep_prob,
                                           FLAGS.keep_prob)
    out, initial_state = tf.nn.dynamic_rnn(cell, key, lk, dtype=tf.float32)

  feat = tf.nn.l2_normalize(initial_state[1], axis=1)

  with tf.variable_scope('Generator'):
    w = tf.get_variable(
      name='embedding',
      shape=[FLAGS.vocab_size, FLAGS.emb_dim],
      initializer=tf.random_uniform_initializer(-0.08, 0.08))
    softmax_w = tf.matrix_transpose(w)
    softmax_b = tf.get_variable('softmax_b', [FLAGS.vocab_size])

    cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.mem_dim)
    if is_training:
      cell = tf.nn.rnn_cell.DropoutWrapper(cell, FLAGS.keep_prob,
                                           FLAGS.keep_prob)
    zero_state = cell.zero_state(FLAGS.batch_size, tf.float32)
    _, state = cell(feat, zero_state)
    init_state = state
    tf.get_variable_scope().reuse_variables()

    state_feed = tf.placeholder(dtype=tf.float32,
                                shape=[None, sum(cell.state_size)],
                                name="state_feed")
    state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)
    input_feed = tf.placeholder(dtype=tf.int64,
                                shape=[None],  # batch_size
                                name="input_feed")
    inputs = tf.nn.embedding_lookup(embedding, input_feed)
    out, state_tuple = cell(inputs, state_tuple)
    tf.concat(axis=1, values=state_tuple, name="state")

    logits = tf.nn.bias_add(tf.matmul(out, softmax_w), softmax_b)
    tower_pred = tf.nn.softmax(logits, name="softmax")
  return tf.concat(init_state, axis=1, name='initial_state')


class Infer:

  def __init__(self, job_dir=FLAGS.job_dir):
    key_inp = tf.placeholder(tf.int32, [None])
    lk = tf.shape(key_inp)[0]
    key = tf.expand_dims(key_inp, axis=0)
    lk = tf.expand_dims(lk, axis=0)
    initial_state_op = _tower_fn(key, lk)

    vocab = vocabulary.Vocabulary(FLAGS.vocab_file)
    self.saver = tf.train.Saver()

    self.key_inp = key_inp
    self.init_state = initial_state_op
    self.vocab = vocab
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    self.sess = tf.Session(config=config)

    self.restore_fn(job_dir)
    self.tf = tf

  def restore_fn(self, checkpoint_path):
    if tf.gfile.IsDirectory(checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    if checkpoint_path:
      self.saver.restore(self.sess, checkpoint_path)
    else:
      self.sess.run(tf.global_variables_initializer())

  def infer(self, key_words):
    vocab = self.vocab
    sess = self.sess
    key_inp = self.key_inp
    initial_state_op = self.init_state
    if key_words.size > 5:
      key_words = key_words[-5:]

    initial_state = sess.run(initial_state_op, feed_dict={key_inp: key_words})

    initial_beam = Caption(
      sentence=[vocab.start_id],
      state=initial_state[0],
      logprob=0.0,
      score=0.0,
      metadata=[""])
    partial_captions = TopN(FLAGS.beam_size)
    partial_captions.push(initial_beam)
    complete_captions = TopN(FLAGS.beam_size)

    # Run beam search.
    for _ in range(FLAGS.max_caption_length - 1):
      partial_captions_list = partial_captions.extract()
      partial_captions.reset()
      input_feed = np.array([c.sentence[-1] for c in partial_captions_list])
      state_feed = np.array([c.state for c in partial_captions_list])

      softmax, new_states = sess.run(
        fetches=["Generator/softmax:0", "Generator/state:0"],
        feed_dict={
          "Generator/input_feed:0": input_feed,
          "Generator/state_feed:0": state_feed,
        })
      metadata = None

      for i, partial_caption in enumerate(partial_captions_list):
        word_probabilities = softmax[i]
        state = new_states[i]
        # For this partial caption, get the beam_size most probable next words.
        words_and_probs = list(enumerate(word_probabilities))
        words_and_probs.sort(key=lambda x: -x[1])
        words_and_probs = words_and_probs[0:FLAGS.beam_size]
        # Each next word gives a new partial caption.
        for w, p in words_and_probs:
          if p < 1e-12:
            continue  # Avoid log(0).
          sentence = partial_caption.sentence + [w]
          logprob = partial_caption.logprob + math.log(p)
          score = logprob
          if metadata:
            metadata_list = partial_caption.metadata + [metadata[i]]
          else:
            metadata_list = None
          if w == vocab.end_id:
            if FLAGS.length_normalization_factor > 0:
              score /= len(sentence) ** FLAGS.length_normalization_factor
            beam = Caption(sentence, state, logprob, score, metadata_list)
            complete_captions.push(beam)
          else:
            beam = Caption(sentence, state, logprob, score, metadata_list)
            partial_captions.push(beam)
      if partial_captions.size() == 0:
        # We have run out of partial candidates; happens when beam_size = 1.
        break

    # If we have no complete captions then fall back to the partial captions.
    # But never output a mixture of complete and partial captions because a
    # partial caption could have a higher score than all the complete captions.
    if not complete_captions.size():
      complete_captions = partial_captions

    captions = complete_captions.extract(sort=True)
    ret = []
    for i, caption in enumerate(captions):
      # Ignore begin and end words.
      sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
      sentence = " ".join(sentence)
      # print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
      ret.append((sentence, math.exp(caption.logprob)))
    return ret
