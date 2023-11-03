S3_ROOT=s3://mm-llm-resources/model-artifacts/sg-mllm
MODEL_DIR=runs/swinl-all_data-q200+50+50-res1024-100k+400k-bs16-polylr-newloss
CKPT_NAME=model_0034999.pth
aws s3 cp $MODEL_DIR/$CKPT_NAME $S3_ROOT/m2f/new_m2f_update/$CKPT_NAME
aws s3 cp $MODEL_DIR/config.yaml $S3_ROOT/m2f/new_m2f_update/config.yaml