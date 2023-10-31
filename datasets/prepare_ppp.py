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

import panoptic_parts as pp
from panoptic_parts import encode_ids, decode_uids
from panoptic_parts.visualization.visualize_label_with_legend import (
    experimental_colorize_label,
)
from panoptic_parts.utils.visualization import set_use_legacy_cpp_parts_colormap

set_use_legacy_cpp_parts_colormap(False)

ppp_spec = pp.specs.dataset_spec.DatasetSpec("ppp_spec.yaml")

natural_name_mapping = {
    'leye': 'left eye',
    'reye': 'right eye',
    'lear': 'left ear',
    'rear': 'right ear',
    'lwing': 'left wing',
    'rwing': 'right wing',
    'lleg': 'left leg',
    'lfoot': 'left foot',
    'rleg': 'right leg',
    'rfoot': 'right foot',
    'lfleg': 'left front leg',
    'lfpa': 'left front paw',
    'rfleg': 'right front leg',
    'rfpa': 'right front paw',
    'lbleg': 'left back leg',
    'lbpa': 'left back paw',
    'rbleg': 'right back leg',
    'rbpa': 'right back paw',
    'lfuleg': 'left front upper leg',
    'lflleg': 'left front lower leg',
    'rfuleg': 'right front upper leg',
    'rflleg': 'right front lower leg',
    'lbuleg': 'left back upper leg',
    'lblleg': 'left back lower leg',
    'rbuleg': 'right back upper leg',
    'rblleg': 'right back lower leg',
    'llarm': 'left lower arm',
    'luarm': 'left upper arm',
    'rlarm': 'right lower arm',
    'ruarm': 'right upper arm',
    'llleg': 'left lower leg',
    'luleg': 'left upper leg',
    'rlleg': 'right lower leg',
    'ruleg': 'right upper leg',
    'lebrow': 'left eyebrow',
    'rebrow': 'right eyebrow',
    'lfho': 'left hoof',
    'rfho': 'right hoof',
    'lbho': 'left back hoof',
    'rbho': 'right back hoof',
    'lhorn': 'left horn',
    'rhorn': 'right horn',
    'hfrontside': 'head front side',
    'hleftside': 'head left side',
    'hrightside': 'head right side',
    'hbackside': 'head back side',
    'hroofside': 'head roof side',
    'cfrontside': 'coach front side',
    'cleftside': 'coach left side',
    'crightside': 'coach right side',
    'cbackside': 'coach back side',
    'croofside': 'coach roof side',
    'fliplate': 'front license plate',
    'bliplate': 'back license plate',
    'frontside': 'front side',
    'backside': 'back side',
    'leftside': 'left side',
    'rightside': 'right side',
    'roofside': 'roof side',
    'leftmirror': 'left mirror',
    'rightmirror': 'right mirror'
}

def get_categories():
    categories = []
    unk = {
        "id": 0,
        "name": "unknown",
        "name_text": "unknown",
        "type": "stuff",
    }
    categories.append(unk)
    for cat in ppp_spec.l_things:
        data = {
            "id": len(categories),
            "name": cat,
            "name_text": cat,
            "type": "thing",
        }
        categories.append(data)
    for cat in ppp_spec.l_stuff:
        data = {
            "id": len(categories),
            "name": cat,
            "name_text": cat,
            "type": "stuff",
        }
        categories.append(data)
    for obj, part in ppp_spec.sid_pid2scene_class_part_class.values():
        if obj == "UNLABELED" or part == "UNLABELED":
            continue
        cat = f"{obj}-{part}"
        part_text = natural_name_mapping.get(part, part)
        text = f"{obj} {part_text}"
        data = {
            "id": len(categories),
            "name": cat,
            "name_text": text,
            "type": "part",
        }
        categories.append(data)
    
    return categories


CATEGORIES = get_categories()
category_to_id = {cat["name"]: cat["id"] for cat in CATEGORIES}

def process_name(name):
    name = name.lower().replace("/", " or ").replace("-", " ").replace("other", "")
    return " ".join(name.strip().split())


def process_sample(img_dir, img_file, label_file_full):
    img_id = int(img_file.replace(".jpg", "").replace("_", ""))
    img_fn = img_file
    img_fn_full = os.path.join(img_dir, img_fn)
    img = Image.open(img_fn_full)

    img_info = create_image_info(img_id, img_fn, img.size)
    annotations = []

    uids = np.array(Image.open(label_file_full), dtype=np.int32)
    uids = encode_ids(
        *decode_uids(
            uids, experimental_dataset_spec=ppp_spec, experimental_correct_range=True
        )
    )

    (
        uids_sem_inst_parts_colored,
        uids_sem_inst_colored,
        uid2color_dct,
    ) = experimental_colorize_label(
        uids,
        sid2color=ppp_spec.sid2scene_color,
        emphasize_instance_boundaries=True,
        return_uid2color=True,
        return_sem_inst=True,
        experimental_deltas=(60, 60, 60),
        experimental_alpha=0.5,
    )

    uids_unique = np.unique(uids)
    things = []
    for uid in uids_unique:
        sid, iid, pid, sid_pid = decode_uids(uid, return_sids_pids=True)
        scene_instance_part_cls = ppp_spec.scene_class_part_class_from_sid_pid(sid_pid)
        if uid >= 1000 and uid <= 99999:
            things.append(scene_instance_part_cls[0])
    
    for uid in uids_unique:
        color = uid2color_dct[uid]
        sid, iid, pid, sid_pid = decode_uids(uid, return_sids_pids=True)
        scene_instance_part_cls = ppp_spec.scene_class_part_class_from_sid_pid(sid_pid)

        ann_id = int(uid)
        if scene_instance_part_cls[0] == "UNLABELED":
            # unlabeled masks
            binary_mask = np.all(uids_sem_inst_colored == color, axis=-1)
            c_id = 0
            ann_info = create_annotation_info(ann_id, img_id, c_id, binary_mask)
            
        elif scene_instance_part_cls[1] != "UNLABELED":
            # part masks
            binary_mask = np.all(uids_sem_inst_parts_colored == color, axis=-1)
            cat = f"{scene_instance_part_cls[0]}-{scene_instance_part_cls[1]}"
            c_id = category_to_id[cat]
            ann_info = create_annotation_info(ann_id, img_id, c_id, binary_mask)
        else: # thing or stuff masks
            if scene_instance_part_cls[0] in things and uid < 100:
                continue
            cat = scene_instance_part_cls[0]
            c_id = category_to_id[cat]
            binary_mask = np.all(uids_sem_inst_colored == color, axis=-1)
            ann_info = create_annotation_info(ann_id, img_id, c_id, binary_mask)
        
        if ann_info is not None:
            annotations.append(ann_info)

    return img_info, annotations


def process_dataset(img_dir, label_dir, split):
    label_files = os.listdir(label_dir)

    inputs = []
    for label_file in label_files:
        img_file = label_file.replace(".tif", ".jpg")
        label_file_full = os.path.join(label_path, label_file)
        inputs.append((img_dir, img_file, label_file_full))

    print(f"Start processing split {split} | num images: {len(inputs)} ...")
    start = time.time()
    pool = mp.Pool(mp.cpu_count())
    jobs = [pool.apply_async(process_sample, args=(*args,)) for args in inputs]
    pool.close()

    all_img_info = []
    all_annotations = []
    for job in tqdm(jobs):
        img_info, annotations = job.get()
        all_img_info.append(img_info)
        all_annotations.extend(annotations)

    print("Finished. time: {:.2f}s".format(time.time() - start))

    coco_data = get_coco_data_json()
    coco_data["images"] = all_img_info
    coco_data["annotations"] = all_annotations

    return coco_data


def get_coco_data_json():
    INFO = {
        "description": "PPP Dataset",
        "url": "",
        "version": "",
        "year": 2022,
        "contributor": "",
        "date_created": datetime.datetime.utcnow().isoformat(" "),
    }

    coco_output = {
        "info": INFO,
        "licenses": [],
        "categories": CATEGORIES,
        "images": [],
        "annotations": [],
    }
    return coco_output


def create_image_info(
    image_id,
    file_name,
    image_size,
    date_captured=datetime.datetime.utcnow().isoformat(" "),
    license_id=1,
    coco_url="",
    flickr_url="",
):
    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[0],
        "height": image_size[1],
        "date_captured": date_captured,
        "license": license_id,
        "coco_url": coco_url,
        "flickr_url": flickr_url,
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


def create_annotation_info(
    annotation_id, image_id, category_id, binary_mask, bounding_box=None
):
    binary_mask_encoded = mask_utils.encode(
        np.asfortranarray(binary_mask.astype(np.uint8))
    )

    area = mask_utils.area(binary_mask_encoded)
    if area < 100:
        # ignore very small masks
        return None

    if bounding_box is None:
        bounding_box = mask_utils.toBbox(binary_mask_encoded)

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
    img_path = "/data/VOCdevkit/VOC2012/JPEGImages"
    for split in ["training", "validation"]:
        label_path = f"/data/pascal_panoptic_parts/labels/{split}"
        coco_data = process_dataset(img_path, label_path, split)
        with open(f"/data/pascal_panoptic_parts/{split}.json", "w") as f:
            json.dump(coco_data, f)