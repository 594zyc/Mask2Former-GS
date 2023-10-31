import contextlib
import datetime
import io
import json
import logging
import numpy as np
import os
import shutil
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from fvcore.common.timer import Timer
from iopath.common.file_io import file_lock
from PIL import Image

from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog

logger = logging.getLogger(__name__)

_ROOT = os.environ.get("DATA_ROOT", "/data")
_SPLIT = {
    "coco_lvis_no_part_train": (
        os.path.join(_ROOT, "coco", "annotations", "panoptic_lvis_merged_no_part_train.json"),
        os.path.join(_ROOT, "coco/images/train2017"),
    ),
    "coco_lvis_paco_part_train": (
        os.path.join(_ROOT, "coco", "annotations", "panoptic_lvis_merged_paco_part_merged_train.json"),
        os.path.join(_ROOT, "coco/images/train2017"),
    ),
    "coco_lvis_val": (
        os.path.join(_ROOT, "coco", "annotations","panoptic_lvis_merged_val.json"),
        os.path.join(_ROOT, "coco/images/val2017"),
    ),
    "entity_train_lr_with_cls": (
        os.path.join(_ROOT, "entityseg", "entityseg_train_lr_with_cls.json"),
        os.path.join(_ROOT, "Adobe_EntitySeg/images_lr"),
    ),
    "entity_train_lr_no_cls": (
        os.path.join(_ROOT, "entityseg", "entityseg_train_lr_no_cls.json"),
        os.path.join(_ROOT, "Adobe_EntitySeg/images_lr"),
    ),
    "ppp_train": (
        os.path.join(_ROOT, "pascal_panoptic_parts", "training.json"),
        os.path.join(_ROOT, "VOCdevkit/VOC2012/JPEGImages"),
    ),
    "ppp_val": (
        os.path.join(_ROOT, "pascal_panoptic_parts", "validation.json"),
        os.path.join(_ROOT, "VOCdevkit/VOC2012/JPEGImages"),
    ),
    "mhp_train": (
        os.path.join(_ROOT, "mhp/LV-MHP-v2", "train.json"),
        os.path.join(_ROOT, "mhp/LV-MHP-v2/train/images"),
    ),
    "mhp_valid": (
        os.path.join(_ROOT, "mhp/LV-MHP-v2", "val.json"),
        os.path.join(_ROOT, "mhp/LV-MHP-v2/val/images"),
    ),
    "textocr_train": (
        os.path.join(_ROOT, "textvqa", "textocr_train.json"),
        os.path.join(_ROOT, "textvqa/images"),
    ),
    "textocr_val": (
        os.path.join(_ROOT, "textvqa", "textocr_val.json"),
        os.path.join(_ROOT, "textvqa/images"),
    )
}


def load_coco_style_json(json_file, image_root, dataset_name):
    """
    Load a json file with COCO's instances annotation format.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    cat_ids = sorted(coco_api.getCatIds())
    cats = coco_api.loadCats(cat_ids)

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': {"count": ..., "size": ...},
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["dataset"] = dataset_name
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            obj = {
                "bbox": anno["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": anno["category_id"],
                "area": anno["area"],
            }
            obj["category_type"] = coco_api.cats[anno["category_id"]]["type"]
            if "textocr" in dataset_name:
                obj["text"] = "text " + anno["text"]
            else:
                obj["text"] = coco_api.cats[anno["category_id"]]["name_text"]

            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )
    return dataset_dicts



def register_instances(name, json_file, image_root):
    """ Register a dataset in COCO's json annotation format for instance segmentation"""
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_coco_style_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    # MetadataCatalog.get(name).set(
    #     json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    # )

for key, (json_file, image_root) in _SPLIT.items():
    register_instances(key, json_file, image_root)