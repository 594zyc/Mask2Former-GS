LOCAL_DATA_ROOT=/data
S3_ROOT=s3://mm-llm-resources/dataset/sgmllm_images


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
    aws s3 cp $LOCAL_DATA_ROOT/$dir $S3_ROOT/$dir
done


# aws s3 cp /home/ec2-user/project/Mask2Former-GS/weights s3://mm-llm-resources/model-artifacts/sg-mllm/model_weights/m2f_weights --recursive
# aws s3 cp s3://mm-llm-resources/model-artifacts/sg-mllm/model_weights/m2f_weights /home/ec2-user/project/Mask2Former-GS/weights --recursive
# aws s3 cp /data/coco/annotations/panoptic_lvis_merged_no_part_train.json s3://mm-llm-resources/dataset/sgmllm_images/coco/annotations/panoptic_lvis_merged_no_part_train.json 