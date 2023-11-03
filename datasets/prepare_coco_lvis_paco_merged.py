#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

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

MIN_PART_AREA_RATIO = 0.03

def load_json(file_name):
    print("Loading json file: {}".format(file_name))
    with open(file_name, "r") as f:
        data = json.load(f)
    return data


def get_obj_and_part_anns(annotations):
    """
    Returns a map between an object annotation ID and 
    (object annotation, list of part annotations) pair.
    """
    ann_id_to_anns = {ann["id"]: (ann, []) for ann in annotations if ann["id"] == ann["obj_ann_id"]}
    for ann in annotations:
        if ann["id"] != ann["obj_ann_id"]:
            ann_id_to_anns[ann["obj_ann_id"]][1].append(ann)
    return ann_id_to_anns


if __name__ == "__main__":

    root_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "/data"))

    # load paco annotations
    paco_train_file = "/data/paco/paco_lvis_v1_train.json"
    paco_val_file = "/data/paco/paco_lvis_v1_val.json"
    paco_val = load_json(paco_val_file)
    paco_train = load_json(paco_train_file)
    # paco_train = {"images": [], "annotations": [], "categories": paco_val["categories"]}

    # create paco indexes
    paco_img_id_to_info = {}
    for img_info in paco_train['images'] + paco_val['images']:
        paco_img_id_to_info[img_info['id']] = img_info
    paco_obj_ann_id_to_anns = get_obj_and_part_anns(paco_train["annotations"])
    paco_obj_ann_id_to_anns.update(get_obj_and_part_anns(paco_val["annotations"]))
    
    paco_img_id_to_anns = {}
    for ann, part_anns in paco_obj_ann_id_to_anns.values():
        img_id = ann["image_id"]
        if img_id not in paco_img_id_to_anns:
            paco_img_id_to_anns[img_id] = []
        paco_img_id_to_anns[img_id].append((ann, part_anns))
    
    paco_category_id_to_info = {}
    for category_info in paco_train['categories']:
        paco_category_id_to_info[category_info['id']] = category_info

    # load coco-lvis merged data
    coco_split = "train"
    coco_ann_dir = os.path.join(root_dir, f"coco/annotations")
    coco_lvis_ann_file = os.path.join(coco_ann_dir, f"panoptic_lvis_merged_{coco_split}.json")
    coco_data = load_json(coco_lvis_ann_file)

    # create indexes
    coco_img_id_to_info = {}
    for img_info in coco_data["images"]:
        coco_img_id_to_info[img_info["id"]] = img_info
    
    coco_category_id_to_info = {}
    for category_info in coco_data["categories"]:
        coco_category_id_to_info[category_info["id"]] = category_info

    coco_img_id_to_anns = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in coco_img_id_to_anns:
            coco_img_id_to_anns[img_id] = []
        cat_info = coco_category_id_to_info[ann["category_id"]]
        coco_img_id_to_anns[img_id].append(ann)
        
    print("Num coco-lvis images:", len(coco_img_id_to_anns))
    print("Num paco images:", len(paco_img_id_to_anns))
    print("Num overlapped images:", len(set(list(coco_img_id_to_anns.keys())).intersection(set(list(paco_img_id_to_anns.keys())))))

    # merge the categories
    new_coco_category_to_info = {}
    for cat_info in coco_category_id_to_info.values():
        cat = cat_info["name"]
        new_coco_category_to_info[cat] = cat_info

    for cat_info in paco_category_id_to_info.values():
        if cat_info["supercategory"] == "OBJECT":
            # lvis category should already in the category list
            continue
        assert cat_info["supercategory"] == "PART"

        cat = cat_info["name"]
        obj, part = cat.split(":")
        obj = obj.replace("_(", " (").replace("_", " ")
        part = part.replace("_(", " (").replace("_", " ")
        name_text = f"{obj} {part}"
        
        new_cat_info = copy.deepcopy(cat_info)
        new_cat_info.update(
            {
                "id": len(new_coco_category_to_info),
                "name": cat,
                "name_text": name_text,
                "type": "part",
                "isthing": 1,
                "ori_paco_id": cat_info["id"],
            }
        )
        new_coco_category_to_info[cat] = new_cat_info

    print("Total categories:", len(new_coco_category_to_info))
    print("From paco:", len([c for c in new_coco_category_to_info.values() if "ori_paco_id" in c]))

    # process inputs

    ori_length = ann_id = len(coco_data["annotations"])
    
    images_no_part = []
    annotations_no_part = []
    images_with_part = []
    annotations_with_part = []
    
    ann_id = len(coco_data["annotations"])
    for img_id, coco_anns in tqdm(coco_img_id_to_anns.items()):
        img_info = coco_img_id_to_info[img_id]
        if img_id not in paco_img_id_to_anns:
            images_no_part.append(img_info)
            annotations_no_part.extend(coco_anns) # note: ann_ids are not continuous anymore
            continue
        
        paco_anns = paco_img_id_to_anns[img_id]
        part_anns_to_add = []
        part_areas_total = 0
        for obj_ann, part_anns in paco_anns:
            for ann in part_anns:
                old_cat_id = ann["category_id"]
                paco_cat = paco_category_id_to_info[old_cat_id]["name"]
                new_cat_id = new_coco_category_to_info[paco_cat]["id"]
                ann["category_id"] = new_cat_id
                ann["id"] = ann_id
                ann_id += 1
                part_anns_to_add.append(ann)
                part_areas_total += ann["area"]
        
        area_ratio = part_areas_total / (img_info["width"] * img_info["height"])
        if  area_ratio < MIN_PART_AREA_RATIO:
            images_no_part.append(img_info)
            annotations_no_part.extend(coco_anns)
        else:
            images_with_part.append(img_info)
            annotations_with_part.extend(coco_anns)
            annotations_with_part.extend(part_anns_to_add)
    

    coco_data_no_parts = coco_data
    coco_data_no_parts.update(
        {
            "images": images_no_part,
            "annotations": annotations_no_part,
            "categories": coco_data["categories"],
        }
    )

    new_categories = [c for c in new_coco_category_to_info.values()]
    coco_data_with_parts = paco_train
    coco_data_with_parts.update(
        {
            "images": images_with_part,
            "annotations": annotations_with_part,
            "categories": new_categories,
        }
    )

    print("Num images:")
    print(" - no parts: {}".format(len(images_no_part)))
    print(" - with parts: {}".format(len(images_with_part)))
    print("Num annotations:")
    print(" - things/stuff {}".format(len(annotations_no_part)))
    print(" - parts: {}".format(len(annotations_with_part)))

    with open(os.path.join(coco_ann_dir, f"panoptic_lvis_merged_no_part_{coco_split}.json"), "w") as f:
        json.dump(coco_data_no_parts, f)
    
    with open(os.path.join(coco_ann_dir, f"panoptic_lvis_merged_paco_part_merged_{coco_split}.json"), "w") as f:
        json.dump(coco_data_with_parts, f)
