export DETECTRON2_DATASETS=/data

MACHINE_RANK=0
NUM_MACHINE=1
DIST_URL="tcp://tcp://xxx.xxx.xxx.xxx:xxxx"
TOTAL_BATCH_SIZE=16
# (need to multiple by NUM_MACHINE)

# related issues:
# https://github.com/facebookresearch/detectron2/issues/3950
# https://github.com/facebookresearch/detectron2/issues/2792
# https://github.com/facebookresearch/detectron2/issues/259

EXP_NAME=swinl-all_data-q200+50+50-res1024-140k+100k-bs16-polylr
python train_net_plus.py \
    --machine-rank $MACHINE_RANK \
    --num-machines $NUM_MACHINE  \
    --dist-url $DIST_URL \
    --num-gpus 8 \
    --config-file weights/coco_panoptic_swinl_100ep/config.yaml \
    OUTPUT_DIR runs/$EXP_NAME \
    WANDB.NAME $EXP_NAME \
    MODEL.WEIGHTS weights/coco_panoptic_swinl_100ep/model_final.pkl \
    DATASETS.TRAIN "('coco_lvis_no_part_train', 'coco_lvis_paco_part_train', 'ppp_train', 'ppp_val', 'entity_train_lr_with_cls', 'entity_train_lr_no_cls', 'textocr_train', 'mhp_train')" \
    SOLVER.IMS_PER_BATCH $TOTAL_BATCH_SIZE \
    SOLVER.BASE_LR 1e-4 \
    INPUT.IMAGE_SIZE 1024 \
    MODEL.META_ARCHITECTURE MaskFormerPlus \
    MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME MultiScaleMaskedTransformerDecoderPlus \
    MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 200 \
    MODEL.MASK_FORMER.NUM_PART_QUERIES 50 \
    MODEL.MASK_FORMER.NUM_TEXT_QUERIES 50 \
    SOLVER.STEPS "(515312, 563812)" \
    SOLVER.MAX_ITER 606250 \
    SOLVER.LR_SCHEDULER_NAME WarmupMultiStepLR \
    TEST.EVAL_PERIOD 100000000 \