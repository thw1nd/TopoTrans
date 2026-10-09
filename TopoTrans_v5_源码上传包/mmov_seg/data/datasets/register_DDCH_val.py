import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

DDCH_CATEGORIES = [
    {"color": [255, 0, 0], "id": 1, "name": "city"},
    {"color": [0, 0, 255], "id": 2, "name": "road"},
    {"color": [0, 255, 0], "id": 3, "name": "farmland"},
    {"color": [255, 255, 0], "id": 4, "name": "forest"},
    {"color": [255, 0, 255], "id": 5, "name": "water"},
]

def _get_DDCH_meta():
    stuff_ids = [k["id"] for k in DDCH_CATEGORIES]
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in DDCH_CATEGORIES]
    stuff_colors = [k["color"] for k in DDCH_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def register_DDCH(root):
    meta = _get_DDCH_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("val", "DDCH/val/rgb_clouds", "DDCH/val/D2masks"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"DDCH_{name}_sem_seg"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )

root = "/yourpath/MM-OVSeg/Dataset"  # 替换为DLRSD数据集的根目录
register_DDCH(root)

