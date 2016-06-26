from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import logging
import re
import sys

sys.path.append('./py_lib/')
from vision import ImageFeatureExtractor
from common_utils import CommonUtiler

logger = logging.getLogger('ExpMscoco')
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)4s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

if __name__ == '__main__':
  flag_ignore_exists = True
  # Path
  model_path = './external/tf_cnn_models/inception_v3.pb'
  mscoco_root = './datasets/ms_coco'
  anno_file_names = ['anno_list_mscoco_trainModelVal_m_RNN.npy',
                     'anno_list_mscoco_crVal_m_RNN.npy',
                     'anno_list_mscoco_test2014.npy']
  feat_dir = './cache/mscoco_image_features/inception_v3'
  
  # Preparations
  cu = CommonUtiler()
  ife = ImageFeatureExtractor(model_path)
  cu.create_dir_if_not_exists(os.path.join(feat_dir, 'train2014'))
  cu.create_dir_if_not_exists(os.path.join(feat_dir, 'test2014'))
  cu.create_dir_if_not_exists(os.path.join(feat_dir, 'val2014'))
  
  # Extract features
  for anno_file_name in anno_file_names:
    anno_path = os.path.join(mscoco_root, 'mscoco_anno_files', anno_file_name)
    annos = np.load(anno_path).tolist()
    for (ind_a, anno) in enumerate(annos):
      image_path = os.path.join(mscoco_root, 'images', anno['file_path'],
          anno['file_name'])
      feat_path = os.path.join(feat_dir, anno['file_path'],
          anno['file_name'].split('.')[0] + '.txt')
          
      if flag_ignore_exists and os.path.exists(feat_path):
        logger.info('%d/%d exists for %s', ind_a+1, len(annos), anno_file_name)
      else:
        try:
          features = ife.extract_features(image_path, flag_from_file=True)
          np.savetxt(feat_path, features, fmt='%.6e')
          logger.info('%d/%d done for %s', ind_a+1, len(annos), anno_file_name)
        except:
          logger.warn('Failed for image %s', image_path)
