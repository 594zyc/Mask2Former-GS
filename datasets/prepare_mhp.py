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

# {'supercategory': 'entity', 'id': 0, 'name': 'unknown', 'c_name': '未知物', 'type': 'thing'},

id_to_category = {
    0: "person",
    1: "Cap/hat",
    2: "Helmet",
    3: "Face",
    4: "Hair",
    5: "Left-arm",
    6: "Right-arm",
    7: "Left-hand",
    8: "Right-hand",
    9: "Protector",
    10: "Bikini/bra",
    11: "Jacket/windbreaker/hoodie",
    12: "Tee-shirt",
    13: "Polo-shirt",
    14: "Sweater",
    15: "Singlet",
    16: "Torso-skin",
    17: "Pants",
    18: "Shorts/swim-shorts",
    19: "Skirt",
    20: "Stockings",
    21: "Socks",
    22: "Left-boot",
    23: "Right-boot",
    24: "Left-shoe",
    25: "Right-shoe",
    26: "Left-highheel",
    27: "Right-highheel",
    28: "Left-sandal",
    29: "Right-sandal",
    30: "Left-leg",
    31: "Right-leg",
    32: "Left-foot",
    33: "Right-foot",
    34: "Coat",
    35: "Dress",
    36: "Robe",
    37: "Jumpsuit",
    38: "Other-full-body-clothes",
    39: "Headwear",
    40: "Backpack",
    41: "Ball",
    42: "Bats",
    43: "Belt",
    44: "Bottle",
    45: "Carrybag",
    46: "Cases",
    47: "Sunglasses",
    48: "Eyewear",
    49: "Glove",
    50: "Scarf",
    51: "Umbrella",
    52: "Wallet/purse",
    53: "Watch",
    54: "Wristband",
    55: "Tie",
    56: "Other-accessary",
    57: "Other-upper-body-clothes",
    58: "Other-lower-body-clothes",
}

thing_ids = {0}
# thing_ids = {0, 1, 2, 40, 41, 42, 44, 45, 46,47, 51, 52, 53, 55}

def get_categories():
    categories = []
    for cid, cat in id_to_category.items():
        data = {
            "id": cid,
            "name": cat,
            "name_text": process_name(cat),
            "type": "thing" if cid in thing_ids else "part"
        }
        categories.append(data)
    return categories

def process_name(name):
    name = name.lower().replace("/", " or ").replace("-", " ").replace("other", "")
    return " ".join(name.strip().split())



def process_sample(img_id, img_dir, ann_files, ann_dir):
    img_fn = f"{img_id}.jpg"
    img_fn_full = os.path.join(img_dir, img_fn)
    img = Image.open(img_fn_full)

    img_info = create_image_info(img_id, img_fn, img.size)
    annotations = []

    for idx, ann_fn in enumerate(ann_files):
        ann_fn_full = os.path.join(ann_dir, ann_fn)
        ann = Image.open(ann_fn_full).convert("RGB")
        ann = np.array(ann)[:, :, 0]
        
        # unique_classes = np.unique(ann.flatten(), axis=0)
        for c in range(59):
            if c == 0:
                binary_mask = ~(ann == c)
            else:
                binary_mask = ann == c
                if binary_mask.sum() == 0:
                    continue

            ann_id = img_id * 1e6 + idx * 1e3 + len(annotations)
            ann_info = create_annotation_info(ann_id, img_id, c, binary_mask)
            annotations.append(ann_info)
    
    return img_info, annotations


def process_dataset(mhp_dir, split):

    img_dir = os.path.join(mhp_dir, split, "images")
    ann_dir = os.path.join(mhp_dir, split, "parsing_annos")


    img_id_to_ann_files = {}
    for ann_fn in os.listdir(ann_dir):
        img_id = int(ann_fn.split("_")[0])
        if img_id not in img_id_to_ann_files:
            img_id_to_ann_files[img_id] = []
        img_id_to_ann_files[img_id].append(ann_fn)
    
    inputs = []
    for img_id, ann_files in img_id_to_ann_files.items():
        inputs.append((img_id, img_dir, ann_files, ann_dir))

    
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
    coco_data["images"] = [sample[0] for sample in all_samples]
    coco_data["annotations"] = [ann for sample in all_samples for ann in sample[1]]

    with open(os.path.join(mhp_dir, split + ".json"), "w") as f:
        json.dump(coco_data, f)


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

    CATEGORIES = get_categories()

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
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


def create_annotation_info(annotation_id, image_id, category_id,  binary_mask, bounding_box=None):

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
        "category_id": category_id,
        "iscrowd": 0,
        "area": area.tolist(),
        "bbox": bounding_box.tolist(),
        "segmentation": segmentation,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    }

    return annotation_info

if __name__ == "__main__":
    dataset_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "/data"), "mhp", "LV-MHP-v2")

    for split in ["val", "train"]:
        process_dataset(dataset_dir, split)