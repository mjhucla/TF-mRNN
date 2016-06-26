# Download and unzip necessary files
# 1. anno list files (Annotation copyright belongs to MS COCO team)
mkdir ./datasets
mkdir ./datasets/ms_coco
wget -O ./datasets/ms_coco/mscoco_anno_files/mscoco_anno_files.zip https://googledrive.com/host/0BztTAiQoAH9CbnRBMXFUZkZVM2c
unzip ./datasets/ms_coco/mscoco_anno_files/mscoco_anno_files.zip -d ./datasets/ms_coco/mscoco_anno_files/
rm ./datasets/ms_coco/mscoco_anno_files/mscoco_anno_files.zip

# 2. inception-v3 model
wget -O ./external/tf_cnn_models/inception_dec_2015.zip https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip
unzip ./external/tf_cnn_models/inception_dec_2015.zip -d ./external/tf_cnn_models/
mv ./external/tf_cnn_models/tensorflow_inception_graph.pb ./external/tf_cnn_models/inception_v3.pb
rm ./external/tf_cnn_models/inception_dec_2015.zip

# 3. the trained mRNN model
wget -O ./trained_model/coco_caption/mrnn_GRU_570K.zip https://googledrive.com/host/0BztTAiQoAH9CVTlGQXJtMGc5d2M
unzip ./trained_model/coco_caption/mrnn_GRU_570K.zip -d ./trained_model/coco_caption/
rm ./trained_model/coco_caption/mrnn_GRU_570K.zip
