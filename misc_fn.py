import tensorflow as tf


def get_len(sequence, end):
  """Gets the length of a generated caption.

  Args:
    sequence: A tensor of size [batch, max_length].
    end: The <EOS> token.

  Returns:
    length: The length of each caption.
  """

  def body(x):
    idx = tf.to_int32(tf.where(tf.equal(x, end)))
    idx = tf.cond(tf.shape(idx)[0] > 0, lambda: idx[0] + 1, lambda: tf.shape(x))
    return idx[0]

  length = tf.map_fn(body, sequence, tf.int32)
  return length


def variable_summaries(var, mask, name):
  """Attaches a lot of summaries to a Tensor.

  Args:
    var: A tensor to summary.
    mask: The mask indicating the valid elements in var.
    name: The name of the tensor in summary.
  """
  var = tf.boolean_mask(var, mask)
  mean = tf.reduce_mean(var)
  tf.summary.scalar('mean/' + name, mean)
  with tf.name_scope('stddev'):
    stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
  tf.summary.scalar('sttdev/' + name, stddev)
  tf.summary.scalar('max/' + name, tf.reduce_max(var))
  tf.summary.scalar('min/' + name, tf.reduce_min(var))
  tf.summary.histogram(name, var)


def transform_grads_fn(grads):
  """Gradient clip."""
  grads, vars = zip(*grads)
  grads, _ = tf.clip_by_global_norm(grads, 10)
  return list(zip(grads, vars))


def crop_sentence(sentence, end):
  """Sentence cropping for logging. Remove the tokens after <EOS>."""
  idx = tf.to_int32(tf.where(tf.equal(sentence, end)))
  idx = tf.cond(tf.shape(idx)[0] > 0, lambda: idx[0] + 1,
                lambda: tf.shape(sentence))
  sentence = sentence[:idx[0]]
  return sentence


def validate_batch_size_for_multi_gpu(batch_size):
  """For multi-gpu, batch-size must be a multiple of the number of GPUs.

  Note that this should eventually be handled by replicate_model_fn
  directly. Multi-GPU support is currently experimental, however,
  so doing the work here until that feature is in place.

  Args:
    batch_size: the number of examples processed in each training batch.

  Raises:
    ValueError: if no GPUs are found, or selected batch_size is invalid.
  """
  from tensorflow.python.client import \
    device_lib  # pylint: disable=g-import-not-at-top

  local_device_protos = device_lib.list_local_devices()
  num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
  if not num_gpus:
    raise ValueError('Multi-GPU mode was specified, but no GPUs '
                     'were found. To use CPU, run without --multi_gpu.')

  remainder = batch_size % num_gpus
  if remainder:
    err = ('When running with multiple GPUs, batch size '
           'must be a multiple of the number of available GPUs. '
           'Found {} GPUs with a batch size of {}; try --batch_size={} instead.'
           ).format(num_gpus, batch_size, batch_size - remainder)
    raise ValueError(err)


def find_obj(sentence, s_mask, classes, scores, num):
  """Computes the object reward for one sentence."""
  shape = tf.shape(sentence)
  sentence = tf.boolean_mask(sentence, s_mask)

  def body(x):
    idx = tf.to_int32(tf.where(tf.equal(sentence, x)))
    idx = tf.cond(tf.shape(idx)[0] > 0, lambda: idx[0, 0],
                  lambda: tf.constant(999, tf.int32))
    return idx

  classes = classes[:num]
  scores = scores[:num]
  ind = tf.map_fn(body, classes, tf.int32)
  mask = tf.not_equal(ind, 999)
  miss, detected = tf.dynamic_partition(scores, tf.to_int32(mask), 2)
  ind = tf.boolean_mask(ind, mask)
  ret = tf.scatter_nd(tf.expand_dims(ind, 1), detected, shape)
  return ret


def obj_rewards(sentence, mask, classes, scores, num):
  """Computes the object reward.

  Args:
    sentence: A tensor of size [batch, max_length].
    mask: The mask indicating the valid elements in sentence.
    classes: [batch, padded_size] int32 tensor of detected objects.
    scores: [batch, padded_size] float32 tensor of detection scores.
    num: [batch] int32 tensor of number of detections.

  Returns:
    rewards: [batch, max_length] float32 tensor of rewards.
  """

  def body(x):
    ret = find_obj(x[0], x[1], x[2], x[3], x[4])
    return ret

  rewards = tf.map_fn(body, [sentence, mask, classes, scores, num], tf.float32)
  return rewards


def random_drop(sentence):
  """Randomly drops some tokens."""
  length = tf.shape(sentence)[0]
  rnd = tf.random_uniform([length]) + 0.9
  mask = tf.cast(tf.floor(rnd), tf.bool)
  sentence = tf.boolean_mask(sentence, mask)
  return sentence


def controlled_shuffle(sentence, d=3.0):
  """Shuffles the sentence as described in https://arxiv.org/abs/1711.00043"""
  length = tf.shape(sentence)[0]
  rnd = tf.random_uniform([length]) * (d + 1) + tf.to_float(tf.range(length))
  _, idx = tf.nn.top_k(rnd, length)
  idx = tf.reverse(idx, axis=[0])
  sentence = tf.gather(sentence, idx)
  return sentence


def _int64_feature(value):
  """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_feature_list(values):
  """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _float_feature(value):
  """Wrapper for inserting an float Feature into a SequenceExample proto."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float_feature_list(values):
  """Wrapper for inserting an float FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_float_feature(v) for v in values])


def _bytes_feature(value):
  """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))
