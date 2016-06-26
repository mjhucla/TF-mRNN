# make necessary directories
mkdir ./cache
mkdir ./cache/mscoco_image_features
wget -O ./cache/mscoco_image_features/vgg_l15.zip https://googledrive.com/host/0BztTAiQoAH9CUEJHclB2ekpiRDg
unzip ./cache/mscoco_image_features/vgg_l15.zip -d ./cache/mscoco_image_features/

