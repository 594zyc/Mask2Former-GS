import functools
import datetime
import json
import copy
from tqdm import tqdm
import numpy as np
import os
import time
from PIL import Image
from pycocotools import mask as mask_utils

def load_json(file_name):
    print("Loading json file: {}".format(file_name))
    with open(file_name, "r") as f:
        data = json.load(f)
    return data

def save_json(data, file_name):
    print("Saving json file: {}".format(file_name))
    with open(file_name, "w") as f:
        json.dump(data, f)

def category_to_text(name: str) -> str:
    name = name.replace("_", " ")
    if name.endswith(" ot"):
        name = name[:-3]
    return name

if __name__ == "__main__":

    root_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "/data"))
    entityv2_img_dir = os.path.join(root_dir, "Adobe_EntitySeg/images")
    entityv2_ann_dir = os.path.join(root_dir, "entityseg")
    
    # load entityv2 annotations 
    for split in ["val", "train"]:
        entityv2_ann_file = os.path.join(entityv2_ann_dir, f"entityseg_{split}_lr.json")
        entityv2_data = load_json(entityv2_ann_file)

        img_id_to_info = {}
        for img_info in entityv2_data['images']:
            img_id_to_info[img_info['id']] = img_info

        img_id_to_anns = {}
        for ann in entityv2_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_id_to_anns:
                img_id_to_anns[img_id] = []
            img_id_to_anns[img_id].append(ann)

        category_id_to_info = {}
        for category_info in entityv2_data['categories']:
            cat_name_text = category_to_text(category_info['name'])
            category_info['name_text'] = cat_name_text
            category_id_to_info[category_info['id']] = category_info
        

        images_with_cls = []
        annotations_with_cls = []
        images_no_cls = []
        annotations_no_cls = []

        for img_id, img_info in img_id_to_info.items():
            anns = img_id_to_anns[img_id]
            if all([ann['category_id'] == 0 for ann in anns]):
                images_no_cls.append(img_info)
                annotations_no_cls.extend(anns)
            else:
                images_with_cls.append(img_info)
                annotations_with_cls.extend(anns)
        
        print(split)
        print(f"images_with_cls: {len(images_with_cls)}")
        print(f"annotations_with_cls: {len(annotations_with_cls)}")
        print(f"images_no_cls: {len(images_no_cls)}")
        print(f"annotations_no_cls: {len(annotations_no_cls)}")

        # save json
        entity_data_with_cls = {
            "images": images_with_cls,
            "annotations": annotations_with_cls,
            "categories": entityv2_data["categories"],
        }
        entity_data_no_cls = {
            "images": images_no_cls,
            "annotations": annotations_no_cls,
            "categories": [category_id_to_info[0]],
        }

        entityv2_with_cls_file = os.path.join(entityv2_ann_dir, f"entityseg_{split}_lr_with_cls.json")
        entityv2_no_cls_file = os.path.join(entityv2_ann_dir, f"entityseg_{split}_lr_no_cls.json")
        save_json(entity_data_with_cls, entityv2_with_cls_file)
        save_json(entity_data_no_cls, entityv2_no_cls_file)