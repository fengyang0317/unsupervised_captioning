"""Detect objects using a model pretrained on OpenImage."""
import multiprocessing
import os

import h5py
import numpy as np
from PIL import Image
from absl import app
from absl import flags
from tqdm import tqdm

from config import TF_MODELS_PATH

flags.DEFINE_string('image_path', None, 'data dir')

flags.DEFINE_integer('num_proc', 1, 'number of process')

flags.DEFINE_integer('num_gpus', 4, 'number of gpus to use')

FLAGS = flags.FLAGS


def load_image_into_numpy_array(image):
  if image.mode != 'RGB':
    image = image.convert('RGB')
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
    (im_height, im_width, 3)).astype(np.uint8)


def initializer():
  import tensorflow as tf
  current = multiprocessing.current_process()
  id = current._identity[0] - 1
  os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % (id % FLAGS.num_gpus)

  model_name = 'faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28'
  path_to_ckpt = (TF_MODELS_PATH + '/research/object_detection/' + model_name
                  + '/frozen_inference_graph.pb')

  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
      od_graph_def.ParseFromString(fid.read())
      tf.import_graph_def(od_graph_def, name='')

  global sess, tensor_dict, image_tensor
  with detection_graph.as_default():
    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(
      allow_growth=True)))
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
      'num_detections', 'detection_boxes', 'detection_scores',
      'detection_classes', 'detection_masks'
    ]:
      tensor_name = key + ':0'
      if tensor_name in all_tensor_names:
        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
          tensor_name)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')


def run(i):
  global sess, tensor_dict, image_tensor
  image_path = FLAGS.image_path + '/' + i.strip()
  image = Image.open(image_path)
  image = load_image_into_numpy_array(image)
  image_np_expanded = np.expand_dims(image, axis=0)
  output_dict = sess.run(tensor_dict,
                         feed_dict={image_tensor: image_np_expanded})
  return i, output_dict


def main(_):
  pool = multiprocessing.Pool(FLAGS.num_proc, initializer)
  with open('data/coco_train.txt', 'r') as f:
    train_images = list(f)
  with open('data/coco_val.txt', 'r') as f:
    val_images = list(f)
  with open('data/coco_test.txt', 'r') as f:
    test_images = list(f)
  all_images = train_images + val_images + test_images
  with h5py.File('data/object.hdf5', 'w') as f:
    for ret in tqdm(pool.imap_unordered(run, all_images),
                    total=len(all_images)):
      name = os.path.splitext(ret[0])[0]
      g = f.create_group(name)
      output_dict = ret[1]
      n = int(output_dict['num_detections'])
      del output_dict['num_detections']
      for k, v in output_dict.items():
        g.create_dataset(k, data=v[0, :n])


if __name__ == '__main__':
  app.run(main)
