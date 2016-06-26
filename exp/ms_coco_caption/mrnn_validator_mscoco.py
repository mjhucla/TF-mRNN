"""TEST for mRNN decoder for MS COCO dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
import os
import numpy as np
import logging
import json
import tensorflow as tf

sys.path.append('./py_lib/')
from common_utils import CommonUtiler
from tf_mrnn_decoder import mRNNDecoder

logger = logging.getLogger('ExpMscoco')
formatter_log = "[%(asctime)s - %(filename)s:line %(lineno)4s] %(message)s"
logging.basicConfig(
    format=formatter_log,
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

flags = tf.flags

# CPU threads
flags.DEFINE_integer("ses_threads", 4, "Tensorflow CPU session threads to use")
# GPU memoery usage
flags.DEFINE_float("gpu_memory_fraction", 0.4, "Fraction of GPU memory to use")
# Model
flags.DEFINE_string("model_root", 
    "./cache/models/mscoco", 
    "root of the tf mRNN model")
flags.DEFINE_string("model_name", "mrnn_GRU_mscoco", "name of the model")
flags.DEFINE_string("eval_stat", 
    "10000 570001 10000", 
    "start_iter step_iter end_iter")
# Vocabulary path
flags.DEFINE_string("vocab_path", 
    "./cache/dctionary/mscoco_mc3_vocab", 
    "path of the vocabulary file for the tf mRNN model")
# Visual feature path
flags.DEFINE_string("vf_dir", 
    "./cache/mscoco_image_features/inception_v3", 
    "directory for the visual feature")
# Validation annotation files
flags.DEFINE_string(
    "anno_files_path", 
    "./datasets/ms_coco/mscoco_anno_files/"
    "anno_list_mscoco_crVal_m_RNN.npy",
    "Validation file annotations, multipy files should be seperated by ':'")
# Beam search size
flags.DEFINE_integer("beam_size", 3, "beam search size")

FLAGS = flags.FLAGS


def main(unused_args):
  # Load model configuration
  cu = CommonUtiler()
  config_path = os.path.join('./model_conf', FLAGS.model_name + '.py')
  config = cu.load_config(config_path)
      
  # Evaluate trained models on val
  decoder = mRNNDecoder(config, FLAGS.model_name, FLAGS.vocab_path,
      gpu_memory_fraction=FLAGS.gpu_memory_fraction)
  for i in xrange(*[int(x) for x in FLAGS.eval_stat.split()]):
    model_path = os.path.join(FLAGS.model_root, FLAGS.model_name, 
        'variables', 'model_%d.ckpt' % i)
    while not os.path.exists(model_path):
      logger.warn('Cannot load model file, sleep 1 hour to retry')
      time.sleep(3600)
    
    decoder.load_model(model_path)
    
    num_decode = 0
    pred_sentences = []
    for anno_file_path in FLAGS.anno_files_path.split(':'):
      annos = np.load(anno_file_path).tolist()
      for anno in annos:
        feat_path = os.path.join(FLAGS.vf_dir, anno['file_path'],
            anno['file_name'].split('.')[0] + '.txt')
        visual_features = np.loadtxt(feat_path)
        sentences = decoder.decode(visual_features, FLAGS.beam_size)
        
        sentence_coco = {}
        sentence_coco['image_id'] = anno['id']
        sentence_coco['caption'] = ' '.join(sentences[0]['words'])
        pred_sentences.append(sentence_coco)
        num_decode += 1
        
        if num_decode % 100 == 0:
          logger.info('%d images are decoded' % num_decode)
          
    pred_path = os.path.join(FLAGS.model_root, FLAGS.model_name, 
        'decode_val_result', 'generated_%d.json' % i)
    result_path = os.path.join(FLAGS.model_root, FLAGS.model_name, 
        'decode_val_result', 'result_%d.txt' % i)
    cu.create_dir_if_not_exists(os.path.dirname(pred_path))
    with open(pred_path, 'w') as fout:
      json.dump(pred_sentences, fout)
    cu.coco_val_eval(pred_path, result_path)
  

if __name__ == "__main__":
  tf.app.run()
