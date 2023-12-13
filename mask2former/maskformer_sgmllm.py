# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple, Optional
import os

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable, get_cfg, CfgNode
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from torchvision.ops import batched_nms

from .config import add_maskformer2_config

@META_ARCH_REGISTRY.register()
class MaskFormerForLLM(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        cfg,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        num_entity_queries: int,
        num_part_queries: int,
        num_text_queries: int,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
        """
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.num_entity_queries = num_entity_queries # for thing/stuff, should be 200
        self.num_part_queries = num_part_queries # for part, should be 50
        self.num_text_queries = num_text_queries # for text, should be 50
        # self.num_queries = 50
        self.num_queries = num_entity_queries + num_part_queries + num_text_queries

        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std


    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        return {
            "cfg": cfg,
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "num_entity_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "num_part_queries": cfg.MODEL.MASK_FORMER.NUM_PART_QUERIES,
            "num_text_queries": cfg.MODEL.MASK_FORMER.NUM_TEXT_QUERIES,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device
    
    @staticmethod
    def get_cfg(
        model_folder: str,
        config_name: str = "config.yaml",
        ckpt_name: str = "model_final.pth",
        use_plus: bool = True,
    ) -> CfgNode:
        # load config from file and command-line arguments
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg, use_plus)
        cfg.merge_from_file(os.path.join(model_folder, config_name))
        cfg.MODEL.META_ARCHITECTURE = "MaskFormerForLLM"
        cfg.MODEL.WEIGHTS = os.path.join(model_folder, ckpt_name)
        cfg.freeze()
        return cfg

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ """
        features = self.backbone(images)
        # the swin backbone cannot work with bf16 somehow
        features = {k:v.to(images) for k,v in features.items()}
        outputs = self.sem_seg_head(features)

        pred_masks = outputs["pred_masks"] # [B,Q,H,W]
        pred_masks = pred_masks.to(images)

        logits = outputs["pred_logits"] # [B,Q,C]

 

        # kept_num = self.num_queries
        # B, Q, H, W = pred_masks.shape
        # C = mask_features.shape[-1]
        # with torch.no_grad():
        #     pred_masks_binary = pred_masks > 0
        #     pred_boxes = batched_mask_to_box(pred_masks_binary).to(mask_features)
        #     scores = pred_masks.clone()
        #     scores[~pred_masks_binary] = 0
        #     scores = scores.sum(dim=[2,3]) # [B,Q]

        #     kept_masks = torch.full((B, kept_num, H, W), -100).to(pred_masks)
        #     kept_feats = torch.full((B, kept_num, C), 0).to(mask_features)
        #     for i in range(B):
        #         boxes_i = pred_boxes[i]
        #         scores_i = scores[i]
        #         indexes_i = batched_nms(boxes_i, scores_i, torch.ones_like(scores_i), 0.8)
        #         indexes_i = indexes_i[:kept_num]
        #         kept_masks[i, :len(indexes_i)] = pred_masks[i, indexes_i]
        #         kept_feats[i, :len(indexes_i)] = mask_features[i, indexes_i]

        #         # import torch.distributed as dist
        #         # if dist.is_initialized() and dist.get_rank() == 0:
        #         # print(f"kept {len(indexes_i)} masks")
        
        # mask_features = kept_feats
        # pred_masks = kept_masks

        return pred_masks, logits