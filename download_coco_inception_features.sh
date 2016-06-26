# make necessary directories
mkdir ./cache
mkdir ./cache/mscoco_image_features
wget -O ./cache/mscoco_image_features/inception_v3.zip https://googledrive.com/host/0BztTAiQoAH9CcXA4TFZLMHNoOUU
unzip ./cache/mscoco_image_features/inception_v3.zip -d ./cache/mscoco_image_features/
