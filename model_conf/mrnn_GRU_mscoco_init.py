class config(object):
  """Config for multimodel RNN training."""
  # Training
  init_scale = 0.05 # variance of the gaussian variable initialization 
  max_grad_norm = 10 # gradient clipping
  learning_rate = 1.0 # inital learning rate (lr)
  lr_decay_keep = 300000 # Num. of iteration that we keep the initial lr
  lr_decay_iter = 10000 # Num. of iteration to apply lr_decay
  lr_decay = 0.85
  num_epoch = 60
  batch_size = 64
  num_iter_save = 10000 # Num. of iteration to save the model
  num_iter_verbose = 100 # Num. of iteration to print the training info.
  buckets = [10, 12, 16, 30]
  
  # Model parameter
  rnn_type = 'GRU' # type of rnn, includes 'GRU' and 'LSTM'
  # different ways to fuze text and visual information,
  # includes 'mrnn' (see visual infor every steps) or 'init' (initiate rnn 
  # hidden state at the first step.
  multimodal_type = 'init'
  # mm_size = 2048 # size of the multimodal layer, only for 'mrnn' type
  # keep_prob_mm = 0.5 # dropout rate of the multimodal layer, only for 'mrnn' type
  rnn_size = 1024 # size of the rnn layer
  emb_size = 512 # size of RNN cell and word emb
  num_rnn_layers = 1
  max_num_steps = 30
  keep_prob_rnn = 0.5 # dropout rate of the rnn output
  keep_prob_emb = 0.5 # dropout rate of the word-embeddings
  vf_size = 2048 # size of the visual feature
  vocab_size = 13691 # size of the vocabulary
