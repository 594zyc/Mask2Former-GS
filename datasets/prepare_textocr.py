#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import functools
import datetime
import json
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

from mask2former.data.datasets.register_entityv2_entity import EntityV2_entity_CATEGORIES




def process_sample(img_id, img_dir, anns):
    img_fn = f"{img_id}.jpg"
    img_fn_full = os.path.join(img_dir, img_fn)
    img = Image.open(img_fn_full)
    img_w, img_h = img.size

    img_info = create_image_info(img_id, img_fn, img.size)
    annotations = []

    for ann in anns:
        ann_id = ann["id"]
        text = ann["utf8_string"]
        if text == ".":
            continue
        
        polygon = np.array(ann["points"])
        rle = mask_utils.frPyObjects([polygon], img_h, img_w)[0]
        binary_mask = mask_utils.decode(rle)
    
        ann_info = create_annotation_info(ann_id, img_id, 1, binary_mask, text)
        annotations.append(ann_info)
    
    if not annotations:
        img_info = None
    return img_info, annotations


def process_dataset(dataset_dir, split):

    img_dir = os.path.join(dataset_dir, "images")
    ann_file = os.path.join(dataset_dir, f"TextOCR_0.1_{split}.json")

    with open(ann_file, "r") as f:
        ann_data = json.load(f)

    ann_id_to_ann = {}
    for ann_ids in ann_data["imgToAnns"].values():
        for ann_id in ann_ids:
            ann = ann_data["anns"][ann_id]
            ann_id_to_ann[ann_id] = ann

    
    inputs = []
    for img_id in ann_data["imgs"]:
        ann_ids = ann_data["imgToAnns"][img_id]
        anns = [ann_id_to_ann[ann_id] for ann_id in ann_ids]
        inputs.append((img_id, img_dir, anns))

    
    print(f"Start processing split {split} | num images: {len(inputs)} ...")
    start = time.time()
    pool = mp.Pool(mp.cpu_count())
    jobs = [pool.apply_async(process_sample, args=(*args,)) for args in inputs]
    pool.close()

    all_samples = []
    for job in tqdm(jobs):
        all_samples.append(job.get())

    print("Finished. time: {:.2f}s".format(time.time() - start))

    coco_data = get_coco_data_json()
    coco_data["images"] = [sample[0] for sample in all_samples if sample[0] is not None]
    coco_data["annotations"] = [ann for sample in all_samples for ann in sample[1]]

    return coco_data


def get_coco_data_json():
    INFO = {
        "description": "MHP v2 Dataset",
        "url": "",
        "version": "",
        "year": 2018,
        "contributor": "",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    LICENSES = []

    categories = [
        {
            "id": 0,
            "name": "text",
            "name_text": "text",
            "type": "text",
        }
    ]

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": categories,
        "images": [],
        "annotations": []
    }
    return coco_output

def create_image_info(image_id, file_name, image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[0],
        "height": image_size[1],
        "date_captured": date_captured,
        "license": license_id,
        "coco_url": coco_url,
        "flickr_url": flickr_url
    }

    return image_info


def binary_mask_to_rle(binary_mask):
    # rle = {'counts': [], 'size': list(binary_mask.shape)}
    # counts = rle.get('counts')
    # for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
    #     if i == 0 and value == 1:
    #         counts.append(0)
    #     counts.append(len(list(elements)))
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def create_annotation_info(annotation_id, image_id, category_id,  binary_mask, text, bounding_box=None):

    binary_mask_encoded = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

    area = mask_utils.area(binary_mask_encoded)
    if area < 1:
        return None

    if bounding_box is None:
        bounding_box =mask_utils.toBbox(binary_mask_encoded)

    segmentation = binary_mask_to_rle(binary_mask)

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 0,
        "iscrowd": 0,
        "area": area.tolist(),
        "bbox": bounding_box.tolist(),
        "segmentation": segmentation,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
        "text": text,
    }

    return annotation_info

if __name__ == "__main__":
    dataset_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "/data"), "textvqa")

    for split in ["val", "train"]:
        coco_data = process_dataset(dataset_dir, split)
        with open(os.path.join(dataset_dir, f"textocr_{split}.json"), "w") as f:
            json.dump(coco_data, f)