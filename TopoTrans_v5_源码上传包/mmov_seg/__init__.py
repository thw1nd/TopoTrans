# Copyright (c) Facebook, Inc. and its affiliates.
try:
    from . import data  # register all new datasets
    from . import modeling
except ImportError as exc:
    if "detectron2" not in str(exc):
        raise

# config
try:
    from .config import add_cat_seg_config
except ImportError as exc:
    if "detectron2" not in str(exc):
        raise
    add_cat_seg_config = None

# dataset loading
try:
    from .data.dataset_mappers.detr_panoptic_dataset_mapper import DETRPanopticDatasetMapper
    from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
        MaskFormerPanopticDatasetMapper,
    )
    from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
        MaskFormerSemanticDatasetMapper,
        MaskFormerSemanticDatasetMapper_test,
    )
except ImportError as exc:
    if "detectron2" not in str(exc):
        raise

# models
try:
    from .test_time_augmentation import SemanticSegmentorWithTTA
except ImportError as exc:
    if "detectron2" not in str(exc):
        raise

#from .MMOV import MMOV   ### If you want to use DINO v1 (ViT-B/16)
try:
    from .MMOVL14 import MMOV   ### If you want to use DINO v3 (ViT-L/14)
except ImportError as exc:
    if "detectron2" not in str(exc):
        raise
from .vision_transformer import vit_base


# from .vision_transformer import * 
# from .vision_transformer import vits
