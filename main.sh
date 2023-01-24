NAME=vit_base_patch16_224
MODEL=vit_base_patch16_224

./extract_feature_timm.py data/imagenet outputs/${NAME}_imagenet_val.pkl ${MODEL} --img_list datalists/imagenet2012_val_list.txt
./extract_feature_timm.py data/imagenet outputs/${NAME}_train_200k.pkl ${MODEL} --img_list datalists/imagenet2012_train_random_200k.txt
./extract_feature_timm.py data/openimage_o outputs/${NAME}_openimage_o.pkl ${MODEL} --img_list datalists/openimage_o.txt
./extract_feature_timm.py data/texture outputs/${NAME}_texture.pkl ${MODEL} --img_list datalists/texture.txt
./extract_feature_timm.py data/inaturalist outputs/${NAME}_inaturalist.pkl ${MODEL}
./extract_feature_timm.py data/imagenet_o outputs/${NAME}_imagenet_o.pkl ${MODEL}