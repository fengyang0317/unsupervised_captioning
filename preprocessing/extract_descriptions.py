"""Extract image descriptions from the downloaded files."""
import cPickle as pkl
import glob
import json
import sys
from multiprocessing import Pool
from unicodedata import normalize

from absl import app
from absl import flags
from tqdm import tqdm

from config import TF_MODELS_PATH

sys.path.insert(0, TF_MODELS_PATH + '/research/im2txt/im2txt')
from data.build_mscoco_data import _process_caption

flags.DEFINE_string('data_dir', 'data/coco', 'data directory')

FLAGS = flags.FLAGS


def main(_):
  s = set()
  files = glob.glob(FLAGS.data_dir + '/*.json')
  files.sort()
  for i in tqdm(files):
    with open(i, 'r') as g:
      data = json.load(g)
      for k, v in data.items():
        for j in v:
          if 'description' in j:
            c = normalize('NFKD', j['description']).encode('ascii', 'ignore')
            c = c.split('\n')
            s.update(c)

  pool = Pool()
  captions = pool.map(_process_caption, list(s))
  pool.close()
  pool.join()
  # There is a sos and eos in each caption, so the actual length is at least 8.
  captions = [i for i in captions if len(i) >= 10]
  print('%s captions parsed' % len(captions))
  with open('data/sentences.pkl', 'w') as f:
    pkl.dump(captions, f)


if __name__ == '__main__':
  app.run(main)
