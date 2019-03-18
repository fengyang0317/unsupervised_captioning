"""Convert the descriptions to tfrecords."""
import cPickle as pkl
import json
import os
import random
import re
import sys
from urllib2 import Request
from urllib2 import urlopen

import tensorflow as tf
from absl import app
from absl import flags
from tqdm import tqdm

from config import TF_MODELS_PATH
from misc_fn import _int64_feature_list

sys.path.insert(0, TF_MODELS_PATH + '/research/im2txt/im2txt')
sys.path.append(TF_MODELS_PATH + '/research')
sys.path.append(TF_MODELS_PATH + '/research/object_detection')
from data.build_mscoco_data import _create_vocab
from inference_utils import vocabulary
from utils import label_map_util

tf.enable_eager_execution()

flags.DEFINE_bool('new_dict', False, 'generate a new dict')

FLAGS = flags.FLAGS


def get_plural(word):
  c = re.compile('Noun</span> <p> \(.*<i>plural</i> ([^\)]+)\)')
  req = Request('https://www.yourdictionary.com/' + word, headers={
    'User-Agent': 'Magic Browser'})
  f = urlopen(req)
  html = f.read()
  f.close()
  html = html.decode('utf-8')
  plural_word = c.findall(html)
  if plural_word:
    plural_word = plural_word[0]
    plural_word = plural_word.lower()
  elif 'Noun</span> <p> (<i>plural only)' in html:
    plural_word = word
  else:
    plural_word = word
    if word[-1] != 's':
      plural_word += 's'
  return plural_word


def get_open_image_categories():
  path_to_labels = (TF_MODELS_PATH + '/research/object_detection/data/'
                                     'oid_bbox_trainable_label_map.pbtxt')
  category_index = label_map_util.create_category_index_from_labelmap(
    path_to_labels,
    use_display_name=True)
  categories = dict([(v['id'], str(v['name'].lower()).split()[-1]) for k, v in
                     category_index.items()])
  category_name = list(set(categories.values()))
  category_name.sort()
  plural_file = 'data/plural_words.json'
  if os.path.exists(plural_file):
    with open(plural_file, 'r') as f:
      plural_dict = json.load(f)
    plural_name = [plural_dict[i] for i in category_name]
  else:
    plural_name = []
    for i in tqdm(category_name):
      plural_name.append(get_plural(i))
    with open(plural_file, 'w') as f:
      json.dump(dict(zip(category_name, plural_name)), f)
  return category_name, plural_name, categories


def parse_key_words(caption, dic):
  key_words = dic.intersection(caption)
  return key_words


def sentence_generator():
  category_name, plural_name, categories = get_open_image_categories()
  replace = dict(zip(plural_name, category_name))
  category_set = set(category_name)

  with open('data/sentences.pkl', 'r') as f:
    captions = pkl.load(f)

  if FLAGS.new_dict:
    _create_vocab(captions)
    with open('data/glove_vocab.pkl', 'r') as f:
      glove = pkl.load(f)
      glove.append('<S>')
      glove.append('</S>')
      glove = set(glove)
    with open(FLAGS.word_counts_output_file, 'r') as f:
      vocab = list(f)
      vocab = [i.strip() for i in vocab]
      vocab = [i.split() for i in vocab]
      vocab = [(i, int(j)) for i, j in vocab if i in glove]
    word_counts = [i for i in vocab if i[0] in category_set or i[1] >= 40]
    words = set([i[0] for i in word_counts])
    for i in category_name:
      if i not in words:
        word_counts.append((i, 0))
    with open(FLAGS.word_counts_output_file, 'w') as f:
      f.write('\n'.join(['%s %d' % (w, c) for w, c in word_counts]))

  vocab = vocabulary.Vocabulary(FLAGS.word_counts_output_file)

  all_ids = dict([(k, vocab.word_to_id(v)) for k, v in categories.items()])
  with open('data/all_ids.pkl', 'w') as f:
    pkl.dump(all_ids, f)

  context = tf.train.Features()
  random.shuffle(captions)
  for c in captions:
    for i, w in enumerate(c):
      if w in replace:
        c[i] = replace[w]
    k = parse_key_words(c, category_set)
    c = [vocab.word_to_id(word) for word in c]
    if c.count(vocab.unk_id) > len(c) * 0.15:
      continue
    k = [vocab.word_to_id(i) for i in k]
    feature_lists = tf.train.FeatureLists(feature_list={
      'key': _int64_feature_list(k),
      'sentence': _int64_feature_list(c)
    })
    sequence_example = tf.train.SequenceExample(
      context=context, feature_lists=feature_lists)
    yield sequence_example.SerializeToString()


def main(_):
  ds = tf.data.Dataset.from_generator(sentence_generator,
                                      output_types=tf.string, output_shapes=())
  tfrec = tf.data.experimental.TFRecordWriter('data/sentence.tfrec')
  tfrec.write(ds)


if __name__ == '__main__':
  app.run(main)
