SPLIT=val2017
python convert_coco_panoptic_to_detection_format.py \
    --input_json_file /data/coco/annotations/panoptic_$SPLIT.json \
    --output_json_file /data/coco/annotations/panoptic_detectron2_$SPLIT.json \
    --segmentations_folder /data/coco/annotations/panoptic_$SPLIT \
    --categories_json_file /data/coco/annotations/panoptic_coco_categories.json \