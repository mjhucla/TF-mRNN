"""Vision module for image feature extraction.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import logging

logger = logging.getLogger('Vision')
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)4s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)


import tensorflow as tf


class ImageFeatureExtractor(object):
  def __init__(self, model_path):
    """Load TensorFlow CNN model."""
    assert os.path.exists(model_path), 'File does not exist %s' % model_path
    self.model_path = model_path
    # load graph
    with tf.gfile.FastGFile(os.path.join(model_path), 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      _ = tf.import_graph_def(graph_def, name='')
    logger.info('Vision graph loaded from %s', model_path)
    # create a session for feature extraction
    self.session = tf.Session()
    self.writer = None
    
  def extract_features(self, image, tensor_name='pool_3:0',
                       flag_from_file=False):
    """Extract image feature from image (numpy array) or from jpeg file."""
    sess = self.session
    feat_tensor = sess.graph.get_tensor_by_name(tensor_name)
    if flag_from_file:
      # image is a path to an jpeg file
      assert os.path.exists(image), 'File does not exist %s' % image
      image_data = tf.gfile.FastGFile(image, 'rb').read()
      features = sess.run(feat_tensor, {'DecodeJpeg/contents:0': image_data})
    else:
      # image is a numpy array with image data
      image_data = image
      features = sess.run(feat_tensor, {'DecodeJpeg:0': image_data})
    
    return np.squeeze(features)
    
  def dump_graph_def(self, log_dir):
    self.writer = tf.train.SummaryWriter(log_dir, self.session.graph)
