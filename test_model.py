import json
import os
import sys

import cv2
from absl import app
from absl import flags
from tqdm import tqdm

from caption_infer import Infer
from config import COCO_PATH

sys.path.insert(0, COCO_PATH)
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

flags.DEFINE_bool('vis', False, 'visulaize')

FLAGS = flags.FLAGS


def main(_):
  infer = Infer()

  with open(COCO_PATH + '/annotations/captions_val2014.json') as g:
    caption_data = json.load(g)
  name_to_id = [(x['file_name'], x['id']) for x in caption_data['images']]
  name_to_id = dict(name_to_id)

  with open('data/coco_test.txt', 'r') as g:
    ret = []
    for name in tqdm(g, total=5000):
      name = name.strip()
      sentences = infer.infer(name)
      cur = {}
      cur['image_id'] = name_to_id[name]
      cur['caption'] = sentences[0][0]
      ret.append(cur)
      if FLAGS.vis:
        im = cv2.imread(FLAGS.data_dir + name)
        print(sentences[0][0])
        cv2.imshow('a', im)
        k = cv2.waitKey()
        if k & 0xff == 27:
          return

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
