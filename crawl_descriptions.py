"""Crawl Shutterstock image descriptions."""
import json
import os
import re
import time
import urllib.request
from multiprocessing import Pool

from absl import app
from absl import flags

flags.DEFINE_string('data_dir', 'data/coco', 'data directory')

flags.DEFINE_integer('num_pages', 1000, 'number of images')

flags.DEFINE_integer('num_processes', 16, 'number of processes')

FLAGS = flags.FLAGS

url = ('https://www.shutterstock.com/search?language=en&image_type=photo&'
       'searchterm=%s&page=%d')
pattern = '<script type="application/ld\+json">(\[.*\])</script>'


class Downloader(object):

  def __init__(self, label):
    self.label = label

  def __call__(self, page_id):
    attempt = 0
    while attempt < 5:
      req = urllib.request.Request(url % (self.label, page_id),
                                   headers={'User-Agent': "Magic Browser"})
      with urllib.request.urlopen(req) as f:
        page = f.read()
      page = page.decode('utf-8')
      obj = re.search(pattern, page)
      if obj is None:
        time.sleep(5)
        attempt += 1
      else:
        break
    if obj is None:
      images = []
    else:
      ret = obj.group(1)
      images = eval(ret)
    return page_id, images


def get_num_pages(label):
  req = urllib.request.Request(
    url % (label, 1),
    headers={'User-Agent': 'Magic Browser'})
  with urllib.request.urlopen(req) as f:
    page = f.read()
  page = page.decode('utf-8')
  obj = re.search('data-max="(\d*)"', page)
  num_pages = int(obj.group(1))
  return num_pages


def download(data_dir, num_pages, id, label):
  output = data_dir + '/%04d.json' % id
  if os.path.exists(output):
    with open(output, 'r') as f:
      images = json.load(f)
    print(label, len(images))
    # print empty pages
    page_nums = [int(k) for k, v in images.items() if len(v) == 0]
    page_nums.sort()
    print(len(page_nums), page_nums)
  else:
    images = {}
    all_pages = get_num_pages(label)
    print(label, all_pages, 'pages available.')
    page_nums = list(range(1, min(num_pages, all_pages) + 1))

  pool = Pool(FLAGS.num_processes)
  pages = pool.map(Downloader(label), page_nums)
  pages = [(str(i[0]), i[1]) for i in pages]
  pool.close()
  pool.join()
  if len(pages) > 0:
    images.update(dict(pages))
    with open(output, 'w') as f:
      json.dump(images, f)


def main(_):
  with open(FLAGS.data_dir + '/coco.names', 'r') as f:
    classes = list(f)
  classes = [i.strip().replace(' ', '+') for i in classes]
  for i, c in enumerate(classes):
    download(FLAGS.data_dir, FLAGS.num_pages, i, c)


if __name__ == '__main__':
  app.run(main)
