LOCAL_DATA_ROOT=/data
S3_ROOT=s3://mm-llm-resources/dataset/sgmllm_images

# download annotations
for dir in \
    coco/annotations/panoptic_lvis_merged_no_part_train.json \
    coco/annotations/panoptic_lvis_merged_paco_part_merged_train.json \
    coco/annotations/panoptic_lvis_merged_val.json \
    entityseg/entityseg_train_lr_with_cls.json \
    entityseg/entityseg_train_lr_no_cls.json \
    pascal_panoptic_parts/training.json \
    pascal_panoptic_parts/validation.json \
    mhp/LV-MHP-v2/val.json \
    mhp/LV-MHP-v2/train.json \
    textvqa/textocr_train.json \
    textvqa/textocr_val.json
do
    aws s3 cp $S3_ROOT/$dir $LOCAL_DATA_ROOT/$dir
done

# download images
for dir in \
    coco/images/train2017.zip \
    coco/images/val2017.zip \
    Adobe_EntitySeg/images_lr/entity_01_11580.zip \
    Adobe_EntitySeg/images_lr/entity_02_11598.zip \
    Adobe_EntitySeg/images_lr/entity_03_10049.zip \
    VOCdevkit/VOC2012.zip \
    mhp/LV-MHP-v2/LV-MHP-v2.zip \
    textvqa/images.zip
do
    aws s3 cp $S3_ROOT/$dir $LOCAL_DATA_ROOT/$dir
done

# unzip images
for dir in \
    coco/images \
    Adobe_EntitySeg/images_lr \
    VOCdevkit \
    mhp/LV-MHP-v2/LV-MHP \
    textvqa
do
    cd $LOCAL_DATA_ROOT/$dir
    echo "unzip $dir"
    UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -q '*.zip'
done


