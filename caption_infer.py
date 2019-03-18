from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from config import TF_MODELS_PATH

sys.path.append(TF_MODELS_PATH + '/research/im2txt/im2txt')
sys.path.append(TF_MODELS_PATH + '/research/slim')
from inference_utils import vocabulary
from inference_utils.caption_generator import Caption
from inference_utils.caption_generator import TopN
from nets import inception_v4

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('job_dir', 'saving', 'job dir')

tf.flags.DEFINE_integer('emb_dim', 512, 'emb dim')

tf.flags.DEFINE_integer('mem_dim', 512, 'mem dim')

tf.flags.DEFINE_integer('batch_size', 1, 'batch size')

tf.flags.DEFINE_string("vocab_file", "data/word_counts.txt",
                       "Text file containing the vocabulary.")

tf.flags.DEFINE_integer('beam_size', 3, 'beam size')

tf.flags.DEFINE_integer('max_caption_length', 20, 'beam size')

tf.flags.DEFINE_float('length_normalization_factor', 0.0, 'l n f')

tf.flags.DEFINE_string('data_dir', None, 'path to all images')

tf.flags.DEFINE_string('inc_ckpt', None, 'InceptionV4 checkpoint path')


def _tower_fn(im, is_training=False):
  with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
    net, _ = inception_v4.inception_v4(im, None, is_training=False)
    net = tf.squeeze(net, [1, 2])

  with tf.variable_scope('Generator'):
    feat = slim.fully_connected(net, FLAGS.mem_dim, activation_fn=None)
    feat = tf.nn.l2_normalize(feat, axis=1)

    embedding = tf.get_variable(
      name='embedding',
      shape=[FLAGS.vocab_size, FLAGS.emb_dim],
      initializer=tf.random_uniform_initializer(-0.08, 0.08))
    softmax_w = tf.matrix_transpose(embedding)
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


def read_image(im):
  """Reads an image."""
  filename = tf.string_join([FLAGS.data_dir, im])
  image = tf.read_file(filename)
  image = tf.image.decode_jpeg(image, 3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize_images(image, [346, 346])
  image = image[23:-24, 23:-24]
  image = image * 2 - 1
  return image


class Infer:

  def __init__(self, job_dir=FLAGS.job_dir):
    im_inp = tf.placeholder(tf.string, [])
    im = read_image(im_inp)
    im = tf.expand_dims(im, 0)
    initial_state_op = _tower_fn(im)

    vocab = vocabulary.Vocabulary(FLAGS.vocab_file)
    self.saver = tf.train.Saver(tf.trainable_variables('Generator'))

    self.im_inp = im_inp
    self.init_state = initial_state_op
    self.vocab = vocab
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    self.sess = tf.Session(config=config)

    inc_saver = tf.train.Saver(tf.global_variables('InceptionV4'))
    self.restore_fn(job_dir)
    inc_saver.restore(self.sess, FLAGS.inc_ckpt)

  def restore_fn(self, checkpoint_path):
    if tf.gfile.IsDirectory(checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    if checkpoint_path:
      self.saver.restore(self.sess, checkpoint_path)
    else:
      self.sess.run(tf.global_variables_initializer())

  def infer(self, im):
    vocab = self.vocab
    sess = self.sess
    im_inp = self.im_inp
    initial_state_op = self.init_state

    initial_state = sess.run(initial_state_op, feed_dict={im_inp: im})

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
        word_probabilities[-1] = 0
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
