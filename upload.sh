S3_ROOT=s3://mm-llm-resources/model-artifacts/sg-mllm
MODEL_DIR=runs/swinl-all_data-q200+50+50-res800-40k-bs16-polylr-entityonly-nosem
CKPT_NAME=model_final.pth
aws s3 cp $MODEL_DIR/$CKPT_NAME $S3_ROOT/m2f/m2f_800res_entityonly_nosem/$CKPT_NAME
aws s3 cp $MODEL_DIR/config.yaml $S3_ROOT/m2f/m2f_800res_entityonly_nosem/config.yaml

# aws s3 cp s3://mm-llm-resources/model-artifacts/sg-mllm/model_weights/m2f_weights/coco_panoptic_swinl_100ep /data/model_weights/coco_panoptic_swinl_100ep --recursive


# aws s3 cp /data/model_weights/coco_panoptic_swinl_100ep s3://mm-llm-resources/model-artifacts/sg-mllm/model_weights/m2f_weights/coco_panoptic_swinl_100ep  --recursive