import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

DDSK_CATEGORIES = [
    {"color": [255, 0, 0], "id": 1, "name": "city"},
    {"color": [0, 0, 255], "id": 2, "name": "road"},
    {"color": [255, 255, 0], "id": 3, "name": "forest"},
    {"color": [255, 0, 255], "id": 4, "name": "water"},
    {"color": [0, 255, 0], "id": 5, "name": "farmland"},
]

def _get_DDSK_meta():
    stuff_ids = [k["id"] for k in DDSK_CATEGORIES]
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in DDSK_CATEGORIES]
    stuff_colors = [k["color"] for k in DDSK_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def register_DDSK(root):
    meta = _get_DDSK_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("val", "DDSK/val/rgb_clouds", "DDSK/val/D2masks"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"DDSK_{name}_sem_seg"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )

root = "/yourpath/MM-OVSeg/Dataset"  # 替换为DLRSD数据集的根目录
register_DDSK(root)

