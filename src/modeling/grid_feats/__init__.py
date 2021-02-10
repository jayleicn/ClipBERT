# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .config import add_attribute_config
from .build_loader import (
    build_detection_train_loader_with_attributes,
    build_detection_test_loader_with_attributes,
)
from .roi_heads import AttributeRes5ROIHeads, AttributeStandardROIHeads
from . import visual_genome