# make necessary directories
mkdir ./cache
mkdir ./cache/mscoco_image_features
wget -O ./cache/mscoco_image_features/vgg_l15.zip http://www.cs.jhu.edu/~jhmao/open_source_data/tf_mrnn/vgg_l15.zip
unzip ./cache/mscoco_image_features/vgg_l15.zip -d ./cache/mscoco_image_features/

