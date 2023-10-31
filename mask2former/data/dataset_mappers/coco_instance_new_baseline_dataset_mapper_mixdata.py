# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Instances, BoxMode, BitMasks, Boxes

import pycocotools.mask as mask_util
import open_clip

from .coco_instance_new_baseline_dataset_mapper import convert_coco_poly_to_mask, build_transform_gen

__all__ = ["COCOInstanceNewBaselineDatasetMapperMixData"]


def annotations_to_instances(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = (
        np.stack(
            [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
        )
        if len(annos)
        else np.zeros((0, 4))
    )
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    # classes = [int(obj["category_id"]) for obj in annos]
    # classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = [obj["text"] for obj in annos]

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        masks = []
        for segm in segms:
            if isinstance(segm, list):
                # polygon
                masks.append(polygons_to_bitmask(segm, *image_size))
            elif isinstance(segm, dict):
                # COCO RLE
                masks.append(mask_util.decode(segm))
            elif isinstance(segm, np.ndarray):
                assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                    segm.ndim
                )
                # mask array
                masks.append(segm)
            else:
                raise ValueError(
                    "Cannot convert segmentation of type '{}' to BitMasks!"
                    "Supported types are: polygons as list[list[float] or ndarray],"
                    " COCO-style RLE as a dict, or a binary segmentation mask "
                    " in a 2D numpy array of shape HxW.".format(type(segm))
                )
        # torch.from_numpy does not support array with negative stride.
        masks = BitMasks(
            torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
        )
        target.gt_masks = masks

    return target

def create_dummy_anno(img_shape):
    dummy_anno = {
        "bbox": [0, 0, 0, 0],
        "bbox_mode": BoxMode.XYXY_ABS,
        "segmentation": np.zeros(img_shape, dtype=np.uint8),
        "text": "nothing",
    }
    return dummy_anno


# This is specifically designed for the COCO dataset.
class COCOInstanceNewBaselineDatasetMapperMixData:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
        num_entity_queries,
        num_part_queries,
        num_text_queries,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[COCOInstanceNewBaselineDatasetMapperMixData] Full TransformGens used in training: {}".format(str(self.tfm_gens))
        )

        self.img_format = image_format
        self.is_train = is_train
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self.num_entity_queries = num_entity_queries
        self.num_part_queries = num_part_queries
        self.num_text_queries = num_text_queries
    
    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "num_entity_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "num_part_queries": cfg.MODEL.MASK_FORMER.NUM_PART_QUERIES,
            "num_text_queries": cfg.MODEL.MASK_FORMER.NUM_TEXT_QUERIES,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            # sorted by area, descending
            annos = sorted(annos, key=lambda x: x["area"], reverse=True)

            grouped_annos = {"entity_instances": [], "part_instances": [], "text_instances": []}

            # Group annos by category type, and keep at most num_<type>_queries GT instances
            for anno in annos:
                if anno['category_type'] in ['thing', 'stuff']:
                    if len(grouped_annos["entity_instances"]) < self.num_entity_queries:
                        grouped_annos["entity_instances"].append(anno)
                elif anno['category_type'] == "part":
                    if len(grouped_annos["part_instances"]) < self.num_part_queries:
                        grouped_annos["part_instances"].append(anno)
                elif anno['category_type'] == "text":
                    if len(grouped_annos["text_instances"]) < self.num_text_queries:
                        grouped_annos["text_instances"].append(anno)
                else:
                    raise ValueError('Unknown category type {}'.format(anno['category_type']))
            

            for group_name, annos in grouped_annos.items():
                if annos:
                    instances = annotations_to_instances(annos, image_shape)
                    # print(instances.gt_classes)
                    instances.gt_classes = self.tokenizer(instances.gt_classes)
                    # After transforms such as cropping are applied, the bounding box may no longer
                    # tightly bound the object. As an example, imagine a triangle object
                    # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
                    # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
                    # the intersection of original bounding box and the cropping box.
                    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                    # Need to filter empty instances first (due to augmentation)
                    instances = utils.filter_empty_instances(instances)
                else:
                    instances = Instances(image_shape, d=[])

                # add dummy annotation to ensure we have at least one instance per group
                if len(instances) == 0:
                    annos = [create_dummy_anno(image_shape)]
                    instances = annotations_to_instances(annos, image_shape)
                    instances.gt_classes = self.tokenizer(instances.gt_classes)
                    # print("add dummy anno: ", group_name)

                dataset_dict[group_name] = instances
        
        assert all(k in dataset_dict for k in ["entity_instances", "part_instances", "text_instances"])
        return dataset_dict
