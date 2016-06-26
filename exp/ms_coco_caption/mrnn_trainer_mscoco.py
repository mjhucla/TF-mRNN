"""mRNN Trainer for MS COCO dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
import os
import numpy as np
import logging
import tensorflow as tf
from tensorflow.python.ops import math_ops

sys.path.append('./py_lib/')
from tf_data_provider import mRNNCocoBucketDataProvider
from common_utils import CommonUtiler
from tf_mrnn_model import mRNNModel

logger = logging.getLogger('ExpMscoco')
formatter_log = "[%(asctime)s - %(filename)s:line %(lineno)4s] %(message)s"
logging.basicConfig(
    format=formatter_log,
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

flags = tf.flags

# CPU threads
flags.DEFINE_integer("ses_threads", 2, "Tensorflow CPU session threads to use")
# Training data
flags.DEFINE_string(
    "anno_files_path", 
    "./datasets/ms_coco/mscoco_anno_files/"
    "anno_list_mscoco_trainModelVal_m_RNN.npy",
    "Training file annotations, multipy files should be seperated by ':'")
# Model paths
flags.DEFINE_string("model_root", 
    "./cache/models/mscoco", 
    "root of the tf mRNN model")
flags.DEFINE_string("model_name", 
    "mrnn_GRU_mscoco", 
    "name of the model")
# Vocabulary path
flags.DEFINE_string("vocab_path", 
    "./cache/dctionary/mscoco_mc3_vocab", 
    "path of the vocabulary file for the tf mRNN model")
# Visual feature path
flags.DEFINE_string("vf_dir", 
    "./cache/mscoco_image_features/inception_v3", 
    "directory for the visual feature")
# Pre-trained model
flags.DEFINE_string("pre_trained_model_path", 
    "", 
    "path of the pre_trained model, if empty will train from scratch.")

FLAGS = flags.FLAGS
    

def run_epoch(session, iters_done, config, models, data_provider, 
    verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  
  # Determine the learning rate with lr decay
  lr_decay_dstep = max(0, 
      (iters_done - config.lr_decay_keep) // config.lr_decay_iter)
  lr_decay = config.lr_decay ** lr_decay_dstep
  for m in models:
    m.assign_lr(session, config.learning_rate * lr_decay)
    
  for step, (ind_buc, x, y, vf, fg, sl) in enumerate(
      data_provider.generate_batches(config.batch_size, config.buckets)):
    # update the lr if necessary
    lr_decay_dstep_cur = max(0, 
        (iters_done + step - config.lr_decay_keep) // config.lr_decay_iter)
    if lr_decay_dstep_cur > lr_decay_dstep:
      lr_decay_dstep = lr_decay_dstep_cur
      lr_decay = config.lr_decay ** lr_decay_dstep
      for m in models:
        m.assign_lr(session, config.learning_rate * lr_decay)
      
    # run forward and backward propgation
    m = models[ind_buc]
    cost, _ = session.run([m.cost, m.train_op],
                          {m.input_data: x,
                           m.targets: y,
                           m.visual_features: vf,
                           m.valid_flags: fg,
                           m.seq_lens: sl})
                           
    costs += cost
    iters += 1

    # print loss if necessary
    if verbose and (iters_done + iters) % config.num_iter_verbose == 0:
      logger.info("Step %d, lr %.3f, model bucket %d(%d)"
                  ": Avg/Cur cost: %.3f/%.3f speed: %.0f sps" %
                  (iters + iters_done, config.learning_rate * lr_decay, 
                   ind_buc, config.buckets[ind_buc],
                   costs / iters, cost, 
                   iters * config.batch_size / (time.time() - start_time)))
      
    # save the current model if necessary
    if (iters_done + iters) % config.num_iter_save == 0:
      models[0].saver.save(session, os.path.join(m.variable_dir, 
          'model_%d.ckpt' % (iters_done + iters)))
      logger.info("Model saved with itereation %d", iters_done + iters)

  return (costs / iters, iters_done + iters)


def main(unused_args):
  # Load model configuration
  cu = CommonUtiler()
  config_path = os.path.join('./model_conf', FLAGS.model_name + '.py')
  config = cu.load_config(config_path)

  # Start model training
  with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(
      intra_op_parallelism_threads=FLAGS.ses_threads)) as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    assert len(config.buckets) >= 1
    assert config.buckets[-1] == config.max_num_steps
    models = []
    with tf.variable_scope("mRNNmodel", reuse=None, initializer=initializer):
      m = mRNNModel(is_training=True,
          num_steps=config.buckets[0], 
          config=config,
          model_name=FLAGS.model_name,
          flag_with_saver=True,
          model_root=FLAGS.model_root)
      models.append(m)
      
    with tf.variable_scope("mRNNmodel", reuse=True):
      for bucket in config.buckets[1:]:
        m = mRNNModel(is_training=True, 
            num_steps=bucket, 
            config=config,
            model_name=FLAGS.model_name,
            model_root=FLAGS.model_root)
        models.append(m)
        
    hdlr = logging.FileHandler(os.path.join(m.model_dir, 'log.txt'))
    hdlr.setLevel(logging.INFO)
    hdlr.setFormatter(logging.Formatter(formatter_log))
    logger.addHandler(hdlr)
    
    if FLAGS.pre_trained_model_path:
      models[0].saver.restore(session, FLAGS.pre_trained_model_path)
      logger.info('Continue to train from %s', FLAGS.pre_trained_model_path)
    else:
      tf.initialize_all_variables().run()

    iters_done = 0
    data_provider = mRNNCocoBucketDataProvider(FLAGS.anno_files_path.split(':'),
        FLAGS.vocab_path, config.vocab_size, FLAGS.vf_dir, config.vf_size)
    for i in range(config.num_epoch):
      train_cost, iters_done = run_epoch(session, iters_done, config, models, 
          data_provider, verbose=True)
      logger.info("Train cost for epoch %d is %.3f" % (i, train_cost))
    
    # Save final copy of the model
    models[0].saver.save(session, os.path.join(m.variable_dir, 
        'model_%d.ckpt' % iters_done))


if __name__ == "__main__":
  tf.app.run()
