import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

DFC25_CATEGORIES = [
    {"color": [128, 0, 0], "id": 1, "name": "Bareland"},
    {"color": [0, 255, 36], "id": 2, "name": "Rangeland"},
    {"color": [148, 148, 148], "id": 3, "name": "Developed space"},
    {"color": [34, 97, 38], "id": 4, "name": "Tree"},
    {"color": [75, 181, 73], "id": 5, "name": "Agriculture Land"},
]

def _get_DFC25_meta():
    stuff_ids = [k["id"] for k in DFC25_CATEGORIES]
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in DFC25_CATEGORIES]
    stuff_colors = [k["color"] for k in DFC25_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def register_DFC25(root):
    meta = _get_DFC25_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("train", "OEM/train/rgb_cloud_thin", "OEM/train/D2masks5class"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"OEMthin_{name}_sem_seg"
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

root = "/yourpath/MM-OVSeg/Dataset"  # Replace with your path
register_DFC25(root)

