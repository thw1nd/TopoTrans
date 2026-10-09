# TopoTrans

TopoTrans v5 source code for multimodal open-vocabulary remote-sensing semantic segmentation. The implementation uses the internal name **MM-OVSeg-MASam V5**.

CLIP provides image/text semantic responses. Adapted SAM2-Hiera features guide semantic graph refinement and coarse-to-fine spatial reconstruction. The released entry points train, evaluate and visualize the v5 model on **ISPRS Potsdam** and **WHU-OPT-SAR**.

## Source layout

- `mmov_seg/mmov_masam_v5.py`: v5 model and margin-guided semantic graph refiner.
- `mmov_seg/modeling/transformer/MMOVPredictorV2.py`: semantic prediction and multiscale guided upsampling.
- `mmov_seg/masam/`: adapted multimodal encoder.
- `train_mmov_masam_potsdam_v5.py`, `train_mmov_masam_whuoptsar_v5.py`: supported training/evaluation entry points.
- `tools/visualize_mmov_masam_*_v5.py`: visualization exporters.
- `configs/`: v5 configurations; `datasets/`: class definitions and split lists.

Some dependencies retain `v2`, `v3`, or `v4` in their filenames because v5 imports or inherits these components. They are required by the released model. Use the v5 entry points below.

## Environment

Use Python 3.10 or newer. Install PyTorch and torchvision appropriate for your machine following the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then:

```bash
pip install -r requirements.txt
```

The local environment used during release preparation has `torch 2.11.0+cu128` and `torchvision 0.26.0+cu128`. The dependency versions above reflect that environment; they are not a claim that every Python/CUDA combination has been tested. Detectron2 is optional for the retained upstream interfaces and is not required for the standalone v5 commands below.

## Data and pretrained encoder

Edit `data_root` in the relevant YAML to point to your dataset. Expected layouts:

```text
data/Potsdam/Images/
  original_train/{rgb,dsm,label}/
  original_test/{rgb,dsm,label}/

data/WHUOPTSAR/original_images/
  opt/
  sar/
  new_lbl/
```

Potsdam pairs RGB `.tif`, DSM `.jpg`, and label `.tif` files by stem. WHU-OPT-SAR pairs optical, SAR and label `.tif` files by filename/stem. Preserve the original class IDs expected by the dataset loaders; colorized visualizations are not training labels. The WHU-OPT-SAR configuration uses the provided train/test ID lists.

Place the SAM 2.1 Hiera Tiny trunk checkpoint at `weights/sam2.1_hiera_tiny.pt`, or edit `masam_weights`. Obtain it from the [official SAM 2 project](https://github.com/facebookresearch/sam2). CLIP ViT-B/16 is loaded by the included CLIP loader and may download its pretrained checkpoint on first use. Dataset images and trained checkpoints are not bundled.

Run all commands from this repository root.

## Training

```bash
python train_mmov_masam_potsdam_v5.py --config configs/RSOVSS_Potsdam_v5.yaml
python train_mmov_masam_whuoptsar_v5.py --config configs/RSOVSS_WHUOPTSAR_v5_mgsr.yaml
```

Adjust `batch_size` and `num_workers` for available memory and operating system. Outputs are saved beneath `outputs/` by default.

## Evaluation

```bash
python train_mmov_masam_potsdam_v5.py --config configs/RSOVSS_Potsdam_v5.yaml --eval-only --resume outputs/topotrans_potsdam_v5/best.pt
python train_mmov_masam_whuoptsar_v5.py --config configs/RSOVSS_WHUOPTSAR_v5_mgsr.yaml --eval-only --resume outputs/topotrans_whuoptsar_v5/best.pt
```

## Visualization

```bash
python tools/visualize_mmov_masam_potsdam_v5.py --config configs/RSOVSS_Potsdam_v5.yaml --resume outputs/topotrans_potsdam_v5/best.pt --output-dir outputs/potsdam_vis --num-samples 16
python tools/visualize_mmov_masam_whuoptsar_v5.py --config configs/RSOVSS_WHUOPTSAR_v5_mgsr.yaml --resume outputs/topotrans_whuoptsar_v5/best.pt --output-dir outputs/whuoptsar_vis
```

## Acknowledgments

This implementation builds on [MM-OVSeg](https://github.com/Jimmyxichen/MM-OVSeg), [CLIP](https://github.com/openai/CLIP), [OpenCLIP](https://github.com/mlfoundations/open_clip), and [SAM 2](https://github.com/facebookresearch/sam2). See `THIRD_PARTY_NOTICES.md` and the preserved upstream README in `docs/`.
