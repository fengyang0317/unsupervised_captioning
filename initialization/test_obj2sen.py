import json
import os
import sys

import tensorflow as tf
from absl import app
from absl import flags
from tqdm import tqdm

from config import COCO_PATH
from eval_obj2sen import parse_image
from sentence_infer import Infer

sys.path.insert(0, COCO_PATH)
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

FLAGS = flags.FLAGS


def main(_):
  infer = Infer()

  with open(COCO_PATH + '/annotations/captions_val2014.json') as g:
    caption_data = json.load(g)
  name_to_id = [(x['file_name'], x['id']) for x in caption_data['images']]
  name_to_id = dict(name_to_id)

  ret = []
  with tf.Graph().as_default(), tf.Session() as sess:
    example = tf.placeholder(tf.string, [])
    name_op, class_op, _ = parse_image(example)
    for i in tqdm(tf.io.tf_record_iterator('data/image_test.tfrec'),
                  total=5000):
      name, classes = sess.run([name_op, class_op], feed_dict={example: i})
      sentences = infer.infer(classes[::-1])
      cur = {}
      cur['image_id'] = name_to_id[name]
      cur['caption'] = sentences[0][0]
      ret.append(cur)

  if os.path.isdir(FLAGS.job_dir):
    out_dir = FLAGS.job_dir
  else:
    out_dir = os.path.split(FLAGS.job_dir)[0]
  out = out_dir + '/test.json'
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

  # print output evaluation scores
  for metric, score in cocoEval.eval.items():
    print('%s: %.3f' % (metric, score))


if __name__ == '__main__':
  app.run(main)
