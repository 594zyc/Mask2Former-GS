#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import functools
import datetime
import json
import copy
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import os
import time
from fvcore.common.download import download
from panopticapi.utils import rgb2id
from PIL import Image
from pycocotools import mask as mask_utils
from itertools import groupby

OVERLAP_THRESHOLD = 0.8

def load_json(file_name):
    print("Loading json file: {}".format(file_name))
    with open(file_name, "r") as f:
        data = json.load(f)
    return data

coco_cls_to_lvis = load_json("coco_class_to_lvis.json")
lvis_syn_to_coco_cls = {info["synset"]:cat for cat, info in coco_cls_to_lvis.items()}


def merge_anns(img_info, coco_anns, lvis_anns, new_catetory_to_info, ann_id):
    if not lvis_anns:
        return coco_anns, ann_id
    
    # convert lvis masks to the rle format
    img_h, img_w = img_info["height"], img_info["width"]
    for ann in lvis_anns:
        ann["segmentation"] = lvis_mask_to_rle(ann["segmentation"], img_h, img_w)
    
    # compute iou between coco and lvis annotations
    coco_masks = [c['segmentation'] for c in coco_anns]
    lvis_masks = [c['segmentation'] for c in lvis_anns]
    iou = mask_utils.iou(coco_masks, lvis_masks, [0] * len(lvis_masks))
    does_overlap = iou.max(axis=1) > OVERLAP_THRESHOLD

    # get all the category names in lvis anns
    lvis_cats = set()
    for ann in lvis_anns:
        cat = get_lvis_category(ann["category_info"])
        lvis_cats.add(cat)
    
    annotations = []
    for idx, ann in enumerate(coco_anns):
        coco_cat = ann["category_info"]["name"]
        # if does_overlap[idx] or coco_cat in lvis_cats:
            # remove coco annotations that overlap with lvis annotations,
            # or the category is in lvis annotations
            # Note: we do not use this because some categories in coco has
            # a different granularity than lvis. For example, bananas in coco
            # is normally annotated as a bunch of bananas, while in lvis it is
            # annotated as individual bananas.
        if does_overlap[idx]:
            # remove coco annotations that overlap with lvis annotations
            continue
        ann["category_id"] = new_catetory_to_info[coco_cat]["id"]
        annotations.append(ann)
    
    # add all annotations from lvis, and update the category and annotation ids
    for ann in lvis_anns:
        cat = get_lvis_category(ann["category_info"])
        ann["category_id"] = new_catetory_to_info[cat]["id"]
        ann["id"] = ann_id
        ann_id += 1
    annotations.extend(lvis_anns)

    # remove category_info
    for a in annotations:
        if "category_info" in a:
            del a["category_info"]
    return annotations, ann_id


def get_lvis_category(cat_info):
    syn = cat_info["synset"]
    if syn in lvis_syn_to_coco_cls:
        cat = lvis_syn_to_coco_cls[syn]
    else:
        cat = cat_info["name"]
    return cat


def lvis_mask_to_rle(segm, height, width):
    rles = mask_utils.frPyObjects(segm, height, width)
    rle = mask_utils.merge(rles)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


if __name__ == "__main__":

    root_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "/data"))

    lvis_train_ann_file = os.path.join(root_dir, "lvis/lvis_v1_train.json")
    lvis_val_ann_file = os.path.join(root_dir, "lvis/lvis_v1_val.json")

    lvis_train = load_json(lvis_train_ann_file)
    lvis_val = load_json(lvis_val_ann_file)
    # lvis_train = {"images": [], "annotations": [], "categories": lvis_val["categories"]}
    
    # create lvis indexes
    lvis_img_id_to_info = {}
    for img_info in lvis_train["images"] + lvis_val["images"]:
        lvis_img_id_to_info[img_info["id"]] = img_info
    
    lvis_category_id_to_info = {}
    for category_info in lvis_train["categories"]:
        lvis_category_id_to_info[category_info["id"]] = category_info

    lvis_img_id_to_anns = {}
    for ann in lvis_train["annotations"] + lvis_val["annotations"]:
        img_id = ann["image_id"]
        if img_id not in lvis_img_id_to_anns:
            lvis_img_id_to_anns[img_id] = []
        cat_info = lvis_category_id_to_info[ann["category_id"]]
        ann["category_info"] = cat_info
        lvis_img_id_to_anns[img_id].append(ann)

    # begin processing
    for coco_split in ["val", "train"]:
        print("Processing {} split".format(coco_split))
        coco_img_dir = os.path.join(root_dir, f"coco/images/{coco_split}2017")
        coco_ann_dir = os.path.join(root_dir, f"coco/annotations")
        coco_ann_file = os.path.join(
            coco_ann_dir, f"panoptic_detectron2_{coco_split}2017.json"
        )

        coco_data = load_json(coco_ann_file)

        # create coco indexes
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
            ann["category_info"] = cat_info
            coco_img_id_to_anns[img_id].append(ann)
            
        print("Num coco images:", len(coco_img_id_to_anns))
        print("Num lvis images:", len(lvis_img_id_to_anns))
        print("Num overlapped images:", len(set(list(coco_img_id_to_anns.keys())).intersection(set(list(lvis_img_id_to_anns.keys())))))

        # merge the categories
        new_coco_category_to_info = {}
        new_coco_category_to_info["unknown"] = {
            "id": 0,
            "name": "unknown",
            "name_text": "unknown",
            "type": "stuff",
            "isthing": 0,
        }
        for cat_info in coco_category_id_to_info.values():
            cat = cat_info["name"]
            new_cat_info = copy.deepcopy(cat_info)
            new_cat_info["ori_coco_id"] = new_cat_info["id"]
            new_cat_info["id"] = len(new_coco_category_to_info)
            new_coco_category_to_info[cat] = new_cat_info

        for cat_info in lvis_category_id_to_info.values():
            cat = cat_info["name"]
            syn = cat_info["synset"]
            if syn in lvis_syn_to_coco_cls:
                coco_cat = lvis_syn_to_coco_cls[syn]
                new_cat_info = new_coco_category_to_info[coco_cat]
                new_cat_info["ori_lvis_id"] = cat_info["id"]
            elif cat in new_coco_category_to_info:
                new_cat_info = new_coco_category_to_info[cat]
                new_cat_info["ori_lvis_id"] = cat_info["id"]
            else:
                new_cat_info = copy.deepcopy(cat_info)
                new_cat_info.update(
                    {
                        "id": len(new_coco_category_to_info),
                        "name": cat,
                        "name_text": cat.replace("_(", " (").replace("_", " "),
                        "type": "thing",
                        "isthing": 1,
                        "ori_lvis_id": cat_info["id"],
                    }
                )
                new_coco_category_to_info[cat] = new_cat_info

        
        print("Total categories:", len(new_coco_category_to_info))
        print("From both:", len([c for c in new_coco_category_to_info.values() if "ori_coco_id" in c and "ori_lvis_id" in c]))
        # for cat_info in new_coco_category_to_info.values():
        #     print(cat_info["id"], cat_info["name_text"], cat_info["from"])

        # process inputs
        start_time = time.time()

        ori_length = ann_id = len(coco_data["annotations"])
        new_annotations = []
        for img_id, coco_anns in tqdm(coco_img_id_to_anns.items()):
            img_info = coco_img_id_to_info[img_id]
            lvis_anns = lvis_img_id_to_anns.get(img_id, [])
            new_anns, ann_id = merge_anns(
                img_info, coco_anns, lvis_anns, new_coco_category_to_info, ann_id
            )
            new_annotations.extend(new_anns)

        new_categories = [c for c in new_coco_category_to_info.values()]

        print("Finished. time: {:.2f}s".format(time.time() - start_time))
        print("Num annotations: {} -> {}".format(ori_length, len(new_annotations)))

        coco_data["categories"] = new_categories
        coco_data["annotations"] = new_annotations
        with open(os.path.join(coco_ann_dir, f"panoptic_lvis_merged_{coco_split}.json"), "w") as f:
            json.dump(coco_data, f)
