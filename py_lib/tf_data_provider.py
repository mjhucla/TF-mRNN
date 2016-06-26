import numpy as np
import os
import logging
import re
import string
import bitarray
import time

from common_utils import CommonUtiler

logger = logging.getLogger('TfDataProvider')
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)4s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)


class Batch(object):
  def __init__(self, batch_size, max_seq_len, vf_size, bos_ind):
    self.batch_size = batch_size
    self.max_seq_len = max_seq_len
    self.vf_size = vf_size
    self.bos_ind = bos_ind
    self.empty()
      
  def empty(self):
    self.x = np.zeros([self.batch_size, self.max_seq_len], dtype=np.int32)
    self.y = np.zeros([self.batch_size, self.max_seq_len], dtype=np.int32)
    self.vf = np.zeros([self.batch_size, self.vf_size], dtype=np.float32)
    self.fg = np.zeros([self.batch_size, self.max_seq_len], dtype=np.float32)
    self.sl = np.zeros([self.batch_size], dtype=np.int32)
    self.num_feed = 0
      
  def feed_and_vomit(self, visual_features, sentence):
    i = self.num_feed
    # feed sentence
    self.x[i, 0] = self.bos_ind
    if len(sentence) > self.max_seq_len - 1:
      self.x[i, 1:] = sentence[:self.max_seq_len-1]
      self.y[i, :self.max_seq_len-1] = sentence[:self.max_seq_len-1]
      self.y[i, self.max_seq_len-1] = self.bos_ind
      self.fg[i, :] = np.ones([self.max_seq_len], dtype=np.float32)
      self.sl[i] = self.max_seq_len
    else:
      l = len(sentence)
      self.x[i, 1:l+1] = sentence
      self.y[i, :l] = sentence
      self.y[i, l] = self.bos_ind
      self.fg[i, :l+1] = np.ones([l+1], dtype=np.float32)
      self.sl[i] = l + 1
    # feed visual feature
    assert visual_features.shape[0] == self.vf_size
    self.vf[i, :] = visual_features
    self.num_feed += 1
    assert self.num_feed <= self.batch_size
    # vomit if necessary
    if self.num_feed == self.batch_size:
      return (self.x, self.y, self.vf, self.fg, self.sl)
    return None


class mRNNCocoBucketDataProvider(object):
  """mRNN TensorFlow Data Provider with Buckets on MS COCO."""
  def __init__(self, anno_files_path, vocab_path, vocab_size, vf_dir, vf_size,
      flag_shuffle=True):
    self.cu = CommonUtiler()
    self.anno_files_path = anno_files_path
    self.vocab_path = vocab_path
    self.vocab, _ = self.cu.load_vocabulary(vocab_path)
    assert len(self.vocab) == vocab_size
    assert self.vocab['<pad>'] == 0
    self.vf_dir = vf_dir
    self.vf_size = vf_size
    self.flag_shuffle = flag_shuffle
    self._load_data()
      
  def generate_batches(self, batch_size, buckets):
    """Return a list generator of mini-batches of training data."""
    # create Batches
    batches = []
    for max_seq_len in buckets:
      batches.append(
          Batch(batch_size, max_seq_len, self.vf_size, self.vocab['<bos>']))
    # shuffle if necessary
    if self.flag_shuffle:
      np.random.shuffle(self._data_pointer)
    # scan data queue
    for ind_i, ind_s in self._data_pointer:
      sentence = self._data_queue[ind_i]['sentences'][ind_s]
      visual_features = self._data_queue[ind_i]['visual_features']
      if len(sentence) >= buckets[-1]:
        feed_res = batches[-1].feed_and_vomit(visual_features, sentence)
        ind_buc = len(buckets) - 1
      else:
        for (ind_b, batch) in enumerate(batches):
          if len(sentence) < batch.max_seq_len:
            feed_res = batches[ind_b].feed_and_vomit(visual_features, sentence)
            ind_buc = ind_b
            break
      if feed_res:
        yield (ind_buc,) + feed_res
        batches[ind_buc].empty()
          
  def _load_data(self, verbose=True):
    logger.info('Loading data')
    vocab = self.vocab
    self._data_queue = []
    self._data_pointer = []
    ind_img = 0
    num_failed = 0
    for anno_file_path in self.anno_files_path:
      annos = np.load(anno_file_path).tolist()
      for (ind_a, anno) in enumerate(annos):
        data = {}
        # Load visual features
        feat_path = os.path.join(self.vf_dir, anno['file_path'],
            anno['file_name'].split('.')[0] + '.txt')
        if os.path.exists(feat_path):
          vf = np.loadtxt(feat_path)
        else:
          num_failed += 1
          continue
        data['visual_features'] = vf
        # Encode sentences
        data['sentences'] = []
        for (ind_s, sentence) in enumerate(anno['sentences']):
          sentence_encode = self.cu.encode_sentence(sentence, vocab, 
              flag_add_bos=False)
          self._data_pointer.append((ind_img, ind_s))
          data['sentences'].append(np.array(sentence_encode))
          
        self._data_queue.append(data)
        ind_img += 1
        if verbose and (ind_a + 1) % 5000 == 0:
          logger.info('Load %d/%d annotation from file %s', ind_a + 1, 
              len(annos), anno_file_path)
        
    logger.info('Load %d images, %d sentences from %d files, %d image failed', 
        len(self._data_queue), len(self._data_pointer), 
        len(self.anno_files_path), num_failed)
