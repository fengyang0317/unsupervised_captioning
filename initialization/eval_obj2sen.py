"""Evaluates the performance of all the checkpoints on validation set."""
import glob
import json
import multiprocessing
import os
import sys

import tensorflow as tf
from absl import app
from absl import flags

from config import COCO_PATH

flags.DEFINE_integer('threads', 1, 'num of threads')

from sentence_infer import Infer

sys.path.insert(0, COCO_PATH)
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

FLAGS = flags.FLAGS


def initializer():
  """Decides which GPU is assigned to a worker.

  If your GPU memory is large enough, you may put several workers in one GPU.
  """
  devices = os.getenv('CUDA_VISIBLE_DEVICES')
  if devices is None:
    devices = []
  else:
    devices = devices.split(',')
  if len(devices) == 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
  else:
    current = multiprocessing.current_process()
    id = (current._identity[0] - 1) % len(devices)
    os.environ['CUDA_VISIBLE_DEVICES'] = devices[id]


def parse_image(serialized):
  """Parses a tensorflow.SequenceExample into an image and detected objects.

  Args:
    serialized: A scalar string Tensor; a single serialized SequenceExample.

  Returns:
    name: A scalar string Tensor containing the image name.
    classes: A 1-D int64 Tensor containing the detected objects.
    scores: A 1-D float32 Tensor containing the detection scores.
  """
  context, sequence = tf.parse_single_sequence_example(
    serialized,
    context_features={
      'image/name': tf.FixedLenFeature([], dtype=tf.string)
    },
    sequence_features={
      'classes': tf.FixedLenSequenceFeature([], dtype=tf.int64),
      'scores': tf.FixedLenSequenceFeature([], dtype=tf.float32),
    })

  name = context['image/name']
  classes = tf.to_int32(sequence['classes'])
  scores = sequence['scores']
  return name, classes, scores


def run(inp):
  out = FLAGS.job_dir + '/val_%s.json' % inp
  if not os.path.exists(out):
    with open(COCO_PATH + '/annotations/captions_val2014.json') as g:
      caption_data = json.load(g)
      name_to_id = [(x['file_name'], x['id']) for x in caption_data['images']]
      name_to_id = dict(name_to_id)

    ret = []
    with tf.Graph().as_default(), tf.Session() as sess:
      example = tf.placeholder(tf.string, [])
      name_op, class_op, _ = parse_image(example)
      infer = Infer(job_dir='%s/model.ckpt-%s' % (FLAGS.job_dir, inp))
      for i in tf.io.tf_record_iterator('data/image_val.tfrec'):
        name, classes = sess.run([name_op, class_op], feed_dict={example: i})
        sentences = infer.infer(classes[::-1])
        cur = {}
        cur['image_id'] = name_to_id[name]
        cur['caption'] = sentences[0][0]
        ret.append(cur)
    with open(out, 'w') as g:
      json.dump(ret, g)

  coco = COCO(COCO_PATH + '/annotations/captions_val2014.json')
  cocoRes = coco.loadRes(out)
  # create cocoEval object by taking coco and cocoRes
  cocoEval = COCOEvalCap(coco, cocoRes)
  # evaluate on a subset of images by setting
  # cocoEval.params['image_id'] = cocoRes.getImgIds()
  # please remove this line when evaluating the full validation set
  cocoEval.params['image_id'] = cocoRes.getImgIds()
  # evaluate results
  cocoEval.evaluate()
  return (inp, cocoEval.eval['CIDEr'], cocoEval.eval['METEOR'],
          cocoEval.eval['Bleu_4'], cocoEval.eval['Bleu_3'],
          cocoEval.eval['Bleu_2'])


def main(_):
  results = glob.glob(FLAGS.job_dir + '/model.ckpt-*')
  results = [os.path.splitext(i)[0] for i in results]
  results = set(results)
  gs_list = [i.split('-')[-1] for i in results]

  pool = multiprocessing.Pool(FLAGS.threads, initializer)
  ret = pool.map(run, gs_list)
  pool.close()
  pool.join()

  ret = sorted(ret, key=lambda x: x[1])
  with open(FLAGS.job_dir + '/cider.json', 'w') as f:
    json.dump(ret, f)
  ret = sorted(ret, key=lambda x: x[2])
  with open(FLAGS.job_dir + '/meteor.json', 'w') as f:
    json.dump(ret, f)
  ret = sorted(ret, key=lambda x: x[3])
  with open(FLAGS.job_dir + '/b4.json', 'w') as f:
    json.dump(ret, f)
  ret = sorted(ret, key=lambda x: x[4])
  with open(FLAGS.job_dir + '/b3.json', 'w') as f:
    json.dump(ret, f)
  ret = sorted(ret, key=lambda x: x[5])
  with open(FLAGS.job_dir + '/b2.json', 'w') as f:
    json.dump(ret, f)
  ret = sorted(ret, key=lambda x: x[3] + x[4])
  with open(FLAGS.job_dir + '/b34.json', 'w') as f:
    json.dump(ret, f)


if __name__ == '__main__':
  app.run(main)
