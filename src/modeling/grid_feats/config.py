# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_attribute_config(cfg):
    """
    Add config for attribute prediction.
    """
    # Whether to have attribute prediction
    cfg.MODEL.ATTRIBUTE_ON = False
    # Maximum number of attributes per foreground instance
    cfg.INPUT.MAX_ATTR_PER_INS = 16
    # ------------------------------------------------------------------------ #
    # Attribute Head
    # -----------------------------------------------------------------------  #
    cfg.MODEL.ROI_ATTRIBUTE_HEAD = CN()
    # Dimension for object class embedding, used in conjunction with 
    # visual features to predict attributes
    cfg.MODEL.ROI_ATTRIBUTE_HEAD.OBJ_EMBED_DIM = 256
    # Dimension of the hidden fc layer of the input visual features
    cfg.MODEL.ROI_ATTRIBUTE_HEAD.FC_DIM = 512
    # Loss weight for attribute prediction, 0.2 is best per analysis
    cfg.MODEL.ROI_ATTRIBUTE_HEAD.LOSS_WEIGHT = 0.2
    # Number of classes for attributes
    cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_CLASSES = 400

    """
    Add config for box regression loss adjustment.
    """
    # Loss weights for RPN box regression
    cfg.MODEL.RPN.BBOX_LOSS_WEIGHT = 1.0
    # Loss weights for R-CNN box regression
    cfg.MODEL.ROI_BOX_HEAD.BBOX_LOSS_WEIGHT = 1.0