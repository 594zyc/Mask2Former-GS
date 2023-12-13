# train
export DETECTRON2_DATASETS=/data/

# pkill -f python

EXP_NAME=swinl-all_data-q200+50+50-res640-200k-bs16-polylr-clsagnostic
python train_net_plus.py \
    --config-file /home/ec2-user/project/Mask2Former-GS/weights/coco_panoptic_swinl_100ep/config.yaml \
    --num-gpus 8 \
    OUTPUT_DIR runs/$EXP_NAME \
    WANDB.NAME $EXP_NAME \
    MODEL.WEIGHTS weights/coco_panoptic_swinl_100ep/model_final.pkl \
    DATASETS.TRAIN "('coco_lvis_no_part_train', 'coco_lvis_paco_part_train', 'ppp_train', 'entity_train_lr_with_cls', 'entity_train_lr_no_cls', 'textocr_train', 'mhp_train')" \
    SOLVER.IMS_PER_BATCH 16 \
    SOLVER.BASE_LR 1e-4 \
    INPUT.IMAGE_SIZE 640 \
    MODEL.META_ARCHITECTURE MaskFormerPlusNoSem \
    MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME MultiScaleMaskedTransformerDecoderPlus \
    MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 200 \
    MODEL.MASK_FORMER.NUM_PART_QUERIES 50 \
    MODEL.MASK_FORMER.NUM_TEXT_QUERIES 50 \
    SOLVER.MAX_ITER 200000 \
    SOLVER.LR_SCHEDULER_NAME WarmupPolyLR \
    TEST.EVAL_PERIOD 100000000000 \
    MODEL.ENTITY_MASK_ONLY True \
    MODEL.MASK_FORMER.NO_OBJECT_WEIGHT 0.5


# 
    # SOLVER.STEPS "(150, 250)" \

# python3 train_net_plus.py \
#     --config-file /home/ec2-user/project/Mask2Former-GS/weights/coco_panoptic_swinl_100ep/config.yaml \
#     --num-gpus 8 \
#     --eval-only \
#     MODEL.WEIGHTS /home/ec2-user/project/Mask2Former-GS/weights/coco_panoptic_swinl_100ep/model_final.pkl \
#     DATASETS.TEST "('coco_2017_val_panoptic',)" \
#     MODEL.MASK_FORMER.TEST.SEMANTIC_ON False \
#     MODEL.MASK_FORMER.TEST.INSTANCE_ON False \
#     MODEL.MASK_FORMER.TEST.PANOPTIC_ON True
