export DETECTRON2_DATASETS=/data-fsx/xiaofeng/sg-mllm

TOTAL_BATCH_SIZE=32 # 2 nodes * 8 gpus * 2 img/gpu

EXP_NAME=swinl-all_data-q200+0+0-res1024-12ep-bs32-steplr-entityonly-nosem
python train_net_plus.py \
    --machine-rank $NODE_RANK \
    --num-machines $NUM_NODES  \
    --dist-url "tcp://$MASTER_ADDR:$MASTER_PORT" \
    --num-gpus 8 \
    --config-file weights/coco_panoptic_swinl_100ep/config.yaml \
    OUTPUT_DIR runs/$EXP_NAME \
    WANDB.NAME $EXP_NAME \
    MODEL.WEIGHTS weights/coco_panoptic_swinl_100ep/model_final.pkl \
    DATASETS.TRAIN "('coco_lvis_no_part_train', 'coco_lvis_paco_part_train', 'entity_train_lr_with_cls', 'entity_train_lr_no_cls')" \
    SOLVER.IMS_PER_BATCH $TOTAL_BATCH_SIZE \
    SOLVER.BASE_LR 1e-4 \
    INPUT.IMAGE_SIZE 1024 \
    MODEL.META_ARCHITECTURE MaskFormerPlusNoSem \
    MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME MultiScaleMaskedTransformerDecoderPlus \
    MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 200 \
    MODEL.MASK_FORMER.NUM_PART_QUERIES 0 \
    MODEL.MASK_FORMER.NUM_TEXT_QUERIES 0 \
    SOLVER.MAX_ITER 55000 \
    SOLVER.LR_SCHEDULER_NAME WarmupPolyLR \
    TEST.EVAL_PERIOD 100000000 \
    MODEL.MASK_FORMER.CLASS_WEIGHT 0.0