# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances

__all__ = [
    "MaskFormerSemanticDatasetMapper",
    "MaskFormerSemanticDatasetMapper_test",
]
import torch.nn as nn
from PIL import Image

class MaskFormerSemanticDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

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
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = 255 #ignore_label
        self.size_divisibility = size_divisibility

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = 255 #meta.ignore_label

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerSemanticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        from PIL import Image
        image = Image.open(dataset_dict["file_name"]).convert("RGB")
        image = np.asarray(image)
        utils.check_image_size(dataset_dict, image)
        
        ###################################################
        image_name = dataset_dict["file_name"]
        from PIL import Image
        if 'PIE_RGBSAR' in image_name:
            if 'rgb_clouds' in image_name:
                sarimage_name = image_name.replace('PIE_RGBSAR/train/rgb_clouds', 'PIE_RGBSAR/train/sar')
            elif 'rgb' in image_name:
                sarimage_name = image_name.replace('PIE_RGBSAR/train/rgb', 'PIE_RGBSAR/train/sar')
        elif 'DDSK' in image_name:
            sarimage_name = image_name.replace('DDSK/train/rgb_clouds', 'DDSK/train/sar')
        elif 'OEM' in image_name:
            if 'rgb_cloud_thin' in image_name:
                sarimage_name = image_name.replace('OEM/train/rgb_cloud_thin', 'OEM/train/imgs')
            elif 'rgb_cloud' in image_name:
                sarimage_name = image_name.replace('OEM/train/rgb_cloud', 'OEM/train/imgs')
        sarimage = Image.open(sarimage_name).convert("RGB")
        sarimage = np.asarray(sarimage)
        utils.check_image_size(dataset_dict, sarimage)
        ###################################################

        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        ###################################################
        # We intentionally do not apply RGB-side dataset mapper transforms to SAR. SAR images contain modality-specific backscatter, speckle, and structural
        # scattering patterns. Extra interpolation-based preprocessing in the mapper may distort these statistics and weaken SAR-specific cues.
        # Instead, SAR is kept at its original resolution during data loading and resized inside the model before being fed into the SAR encoder. 
        # This provides a modality-specific resizing strategy and avoids redundant resampling.
        aug_input0 = T.AugInput(sarimage, sem_seg=sem_seg_gt)
        sarimage = aug_input0.image
        sarimage = torch.as_tensor(np.ascontiguousarray(sarimage.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
        # import ipdb; ipdb.set_trace()
        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            # The ori_size is not the real original size, but size before padding
            dataset_dict['ori_size'] = image_size
            padding_size = [
                0,
                self.size_divisibility - image_size[1], # w: (left, right)
                0,
                self.size_divisibility - image_size[0], # h: 0,(top, bottom)
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            ################################################################
            sarimage_size = (sarimage.shape[-2], sarimage.shape[-1])
            padding_size0 = [
                0,
                self.size_divisibility - sarimage_size[1],
                0,
                self.size_divisibility - sarimage_size[0],
            ]
            sarimage = F.pad(sarimage, padding_size0, value=128).contiguous()
            ################################################################
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image
        ############################################333
        dataset_dict["sarimage"] = sarimage
        ###############################################
        # print('#########################################################################################')
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

        return dataset_dict

class MaskFormerSemanticDatasetMapper_test:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=False,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=False):
        # Only resize is needed!
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]

        dataset_names = cfg.DATASETS.TEST
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = 255 #meta.ignore_label

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
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
        # image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        from PIL import Image
        image = Image.open(dataset_dict["file_name"]).convert("RGB")
        image = np.asarray(image)
        utils.check_image_size(dataset_dict, image)
        
        ###################################################
        image_name = dataset_dict["file_name"]
        from PIL import Image
        if 'PIE_RGBSAR' in image_name:
            if 'rgb_clouds' in image_name:
                sarimage_name = image_name.replace('PIE_RGBSAR/val/rgb_clouds', 'PIE_RGBSAR/val/sar')
            elif 'rgb' in image_name:
                sarimage_name = image_name.replace('PIE_RGBSAR/val/rgb', 'PIE_RGBSAR/val/sar')
        elif 'DDSK' in image_name:
            sarimage_name = image_name.replace('DDSK/val/rgb_clouds', 'DDSK/val/sar')
        elif 'DDCH' in image_name:
            sarimage_name = image_name.replace('DDCH/val/rgb_clouds', 'DDCH/val/sar')
        elif 'OEM' in image_name:
            if 'rgb_cloud_thin' in image_name:
                sarimage_name = image_name.replace('OEM/val/rgb_cloud_thin', 'OEM/val/imgs')
            elif 'rgb_cloud' in image_name:
                sarimage_name = image_name.replace('OEM/val/rgb_cloud', 'OEM/val/imgs')
        sarimage = Image.open(sarimage_name).convert("RGB")
        sarimage = np.asarray(sarimage)
        utils.check_image_size(dataset_dict, sarimage)
        ###################################################
        #print("Image Name: ", image_name," ",sarimage_name)
        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        ###################################################
        aug_input0 = T.AugInput(sarimage, sem_seg=sem_seg_gt)
        sarimage = aug_input0.image
        sarimage = torch.as_tensor(np.ascontiguousarray(sarimage.transpose(2, 0, 1)))

        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
        # import ipdb; ipdb.set_trace()
        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            # The ori_size is not the real original size, but size before padding
            dataset_dict['ori_size'] = image_size
            padding_size = [
                0,
                self.size_divisibility - image_size[1], # w: (left, right)
                0,
                self.size_divisibility - image_size[0], # h: 0,(top, bottom)
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            ################################################################
            sarimage_size = (sarimage.shape[-2], sarimage.shape[-1])
            padding_size0 = [
                0,
                self.size_divisibility - sarimage_size[1],
                0,
                self.size_divisibility - sarimage_size[0],
            ]
            sarimage = F.pad(sarimage, padding_size0, value=128).contiguous()
            ################################################################
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image
        ############################################
        dataset_dict["sarimage"] = sarimage
        ############################################
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

        return dataset_dict