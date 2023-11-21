S3_ROOT=s3://mm-llm-resources/model-artifacts/sg-mllm
MODEL_DIR=m2f_all_100k
aws s3 cp $S3_ROOT/m2f/$MODEL_DIR runs/$MODEL_DIR --recursive