"""Decoder (sentence generator) for the trained mRNN model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import logging
import copy

from common_utils import CommonUtiler
from tf_mrnn_model import mRNNModel

logger = logging.getLogger('TfMrnnDecoder')
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)4s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)


import tensorflow as tf


class mRNNDecoder(object):
  """The sentence decoder (generator) for mRNNModel."""

  def __init__(self, config, model_name, vocab_path,
               ses_threads=2,
               gpu_memory_fraction=1.0):
    self.cu = CommonUtiler()
    self.config = copy.deepcopy(config)
    self.config.batch_size = 1
    self.model_path = None
    self.model_name = model_name
    self.flag_load_model = False
    self.vocab_path = vocab_path
    self.vocab, self.rev_vocab = self.cu.load_vocabulary(vocab_path)
    
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=gpu_memory_fraction)
    self.session = session = tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=ses_threads, 
        gpu_options=gpu_options))
    
    with tf.variable_scope("mRNNmodel", reuse=None):
      self.model_init = mRNNModel(
          is_training=False,
          num_steps=1, 
          config=self.config,
          model_name=self.model_name,
          flag_with_saver=True)
    
    with tf.variable_scope("mRNNmodel", reuse=True):
      self.model_cont = mRNNModel(
          is_training=False,
          num_steps=1, 
          config=self.config,
          model_name=self.model_name,
          flag_with_saver=False,
          flag_reset_state=True)
          
  def load_model(self, model_path):
    self.model_init.saver.restore(self.session, model_path)
    self.flag_load_model = True
    self.model_path = model_path
    logger.info('Load model from %s', model_path)
    
  def decode(self, visual_features, beam_size, max_steps=30):
    """Decode an image with a sentences."""
    assert visual_features.shape[0] == self.config.vf_size
    assert self.flag_load_model, 'Must call local_model first'
    vocab = self.vocab
    rev_vocab = self.rev_vocab
    
    # Initilize beam search variables
    # Candidate will be represented with a dictionary
    #   "indexes": a list with indexes denoted a sentence; 
    #   "words": word in the decoded sentence without <bos>
    #   "score": log-likelihood of the sentence
    #   "state": RNN state when generating the last word of the candidate
    good_sentences = [] # store sentences already ended with <bos>
    cur_best_cand = [] # store current best candidates
    highest_score = 0.0 # hightest log-likelihodd in good sentences
    
    # Get the initial logit and state
    logit_init, state_init = self.get_logit_init(visual_features)
    logit_init = np.squeeze(logit_init)
    assert logit_init.shape[0] == self.config.vocab_size and len(
        logit_init.shape) == 1
    logit_init = self.cu.softmax(logit_init)
    logit_init_order = np.argsort(-logit_init)
    for ind_b in xrange(beam_size):
      cand = {}
      cand['indexes'] = [logit_init_order[ind_b]]
      cand['score'] = -np.log(logit_init[logit_init_order[ind_b]])
      cand['state'] = state_init
      cur_best_cand.append(cand)
      
    # Expand the current best candidates until max_steps or no candidate
    for i in xrange(max_steps):
      # move candidates end with <bos> to good_sentences or remove it
      cand_left = []
      for cand in cur_best_cand:
        if len(good_sentences) > beam_size and cand['score'] > highest_score:
          continue # No need to expand that candidate
        if cand['indexes'][-1] == vocab['<bos>']:
          good_sentences.append(cand)
          highest_score = max(highest_score, cand['score'])
        else:
          cand_left.append(cand)
      cur_best_cand = cand_left
      if not cur_best_cand:
        break
      # expand candidate left
      cand_pool = []
      for cand in cur_best_cand:
        logit, state = self.get_logit_cont(cand['state'], cand['indexes'][-1],
            visual_features)
        logit = np.squeeze(logit)
        logit = self.cu.softmax(logit)
        logit_order = np.argsort(-logit)
        for ind_b in xrange(beam_size):
          cand_e = copy.deepcopy(cand)
          cand_e['indexes'].append(logit_order[ind_b])
          cand_e['score'] -= np.log(logit[logit_order[ind_b]])
          cand_e['state'] = state
          cand_pool.append(cand_e)
      # get final cand_pool
      cur_best_cand = sorted(cand_pool, key=lambda cand: cand['score'])
      cur_best_cand = self.cu.truncate_list(cur_best_cand, beam_size)
      
    # Add candidate left in cur_best_cand to good sentences
    for cand in cur_best_cand:
      if len(good_sentences) > beam_size and cand['score'] > highest_score:
        continue
      if cand['indexes'][-1] != vocab['<bos>']:
        cand['indexes'].append(vocab['<bos>'])
      good_sentences.append(cand)
      highest_score = max(highest_score, cand['score'])
      
    # Sort good sentences and return the final list
    good_sentences = sorted(good_sentences, key=lambda cand: cand['score'])
    good_sentences = self.cu.truncate_list(good_sentences, beam_size)
    for sentence in good_sentences:
      sentence['words'] = self.cu.decode_sentence(
          sentence['indexes'], vocab, rev_vocab)
    
    return good_sentences
    
  def get_logit_init(self, visual_features):
    """Use the model to get initial logit"""
    m = self.model_init
    session = self.session
    vocab = self.vocab
    config = self.config
    
    x = np.zeros([1, 1], dtype=np.int32)
    vf = np.zeros([1, config.vf_size], dtype=np.float32)
    fg = np.ones([1, 1], dtype=np.float32)
    sl = np.ones([1], dtype=np.int32)
    vf[0, :] = visual_features
    x[0] = vocab['<bos>']
    
    logit, state = session.run([m.logit, m.final_state],
                               {m.input_data: x,
                                m.visual_features: vf,
                                m.valid_flags: fg,
                                m.seq_lens: sl})
                              
    return (logit, state)
    
  def get_logit_cont(self, state_prev, index_word, visual_features):
    """Use the model to get continued logit"""
    m = self.model_cont
    session = self.session
    config = self.config
    
    x = np.zeros([1, 1], dtype=np.int32)
    vf = np.zeros([1, config.vf_size], dtype=np.float32)
    fg = np.ones([1, 1], dtype=np.float32)
    sl = np.ones([1], dtype=np.int32)
    vf[0, :] = visual_features
    x[0] = index_word
    
    logit, state = session.run([m.logit, m.final_state],
                               {m.input_data: x,
                                m.visual_features: vf,
                                m.valid_flags: fg,
                                m.seq_lens: sl,
                                m.initial_state: state_prev})
                              
    return (logit, state)
