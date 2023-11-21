# train
export DETECTRON2_DATASETS=/data/

# pkill -f python
# sleep 2
EXP_NAME=swinl-all_data-q200+0+0-res640-2k-bs16-polylr-entityonly-nosem
python train_net_plus.py \
    --config-file /home/ec2-user/project/Mask2Former-GS/weights/coco_panoptic_swinl_100ep/config.yaml \
    --num-gpus 8 \
    OUTPUT_DIR runs/$EXP_NAME \
    WANDB.NAME $EXP_NAME \
    MODEL.WEIGHTS weights/coco_panoptic_swinl_100ep/model_final.pkl \
    DATASETS.TRAIN "('coco_lvis_no_part_train', 'coco_lvis_paco_part_train', 'entity_train_lr_with_cls', 'entity_train_lr_no_cls')" \
    SOLVER.IMS_PER_BATCH 16 \
    SOLVER.BASE_LR 1e-4 \
    INPUT.IMAGE_SIZE 640 \
    MODEL.META_ARCHITECTURE MaskFormerPlus \
    MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME MultiScaleMaskedTransformerDecoderPlus \
    MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 200 \
    MODEL.MASK_FORMER.NUM_PART_QUERIES 0 \
    MODEL.MASK_FORMER.NUM_TEXT_QUERIES 0 \
    SOLVER.MAX_ITER 2000 \
    SOLVER.LR_SCHEDULER_NAME WarmupPolyLR \
    TEST.EVAL_PERIOD 100000000000 \
    MODEL.MASK_FORMER.CLASS_WEIGHT 0.0



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
