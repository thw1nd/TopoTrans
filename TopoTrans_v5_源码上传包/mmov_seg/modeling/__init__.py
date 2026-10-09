# Copyright (c) Facebook, Inc. and its affiliates.
try:
    from .backbone.swin import D2SwinTransformer
    from .heads.MMOVHead import MMOVHead
except ImportError as exc:
    if "detectron2" not in str(exc):
        raise
