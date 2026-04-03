# AIFlash - Baseline

[🇨🇳 中文说明 (Chinese)](README_CN.md)

---

## 🇬🇧 English

AIFlash baseline code based on [Restormer](https://github.com/swz30/Restormer).

### 🎯 Challenge Motivation

Despite significant advances in deep learning for image restoration, existing models still struggle to balance noise removal, detail preservation, and light/color restoration in real-world low-light portrait scenarios. **AI Flash Portrait** aims to generate natural, clean, and realistic portraits while maintaining comfortable background appearance and accurate colors.

### 🏆 Challenge Objectives

This challenge provides a platform for industrial and academic participants to test and evaluate algorithms and models in real-world imaging scenarios. The challenge objectives include:

- Establishing a benchmark for low-light portrait restoration in real scenes, covering both objective and subjective evaluation;
- Promoting the development and practical application of highly robust algorithms;
- Encouraging solutions that excel in both portrait and overall scene quality.

---

### ⚠️ Data Description

| Data | Resolution |
|------|------------|
| Input Image | **1K** |
| Ground Truth (GT) | **1K** |
| Submission Result | **1K** |

> **Note**: The current baseline resizes images to **768** for training. Participants can adjust the training resolution based on their GPU memory. The final inference result must be in **1K resolution**.

---


### 📥 Dataset Download

The full dataset is now publicly available (all stages, with GT) on Hugging Face. The original competition page is kept for reference:

- Public dataset: [https://huggingface.co/datasets/zsn1434/AI_Flash_Portrait/tree/main](https://huggingface.co/datasets/zsn1434/AI_Flash_Portrait)
- Competition page: [https://www.codabench.org/competitions/12885/](https://www.codabench.org/competitions/12885/)

| Stage | Dataset | Status | Description |
|-------|---------|--------|-------------|
| Stage 1 | Training Set | ✅ Available | Training images (input + GT pairs + person mask) |
| Stage 2 | Test Set A | ✅ Available (public, with GT) | Test images and GT publicly released on Hugging Face |
| Stage 3 | Test Set B | ✅ Available (public, with GT) | Test images and GT publicly released on Hugging Face |


#### Directory Structure

After downloading, please organize the data as follows:

```
AI_Flash-BaseLine/
├── datasets/
│   ├── train/
│   │   ├── input/          # Training input images (low-light)
│   │   │   ├── 0001.jpg
│   │   │   ├── 0002.jpg
│   │   │   └── ...
│   │   ├── gt/             # Training ground truth images
│   │   │   ├── 0001.jpg
│   │   │   ├── 0002.jpg
│   │   │   └── ...
│   │   └── mask_personmask/    # Person segmentation masks (if used)
│   │       ├── 0001.jpg
│   │       ├── 0002.jpg
│   │       └── ...
│   └── test/
│       ├── input/          # Test input images
│       │   ├── 0001.jpg
│       │   ├── 0002.jpg
│       │   └── ...
│       └── mask_personmask/    # Person segmentation masks (if used)
│           ├── 0001.jpg
│           ├── 0002.jpg
│           └── ...
├── configs/
├── basicsr/
└── ...
```

---


### Environment Setup

#### 1. Create Conda Environment

```bash
conda create -n aiflash python=3.10 -y
conda activate aiflash
```

#### 2. Install PyTorch (CUDA 12.1)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> For other CUDA versions, please refer to [PyTorch Official Website](https://pytorch.org/get-started/locally/)

#### 3. Install Other Dependencies

```bash
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb "numpy<2" pyyaml requests scipy tensorboard yapf lpips
```

#### 4. Install BasicSR

```bash
cd AI_Flash-BaseLine
pip install -e .
```

---

### Training

```bash
python basicsr/train.py -opt configs/flash.yaml
```

---

### Inference

```bash
python inference.py \
    --config configs/flash.yaml \
    --weights pth/net_g_xxx.pth \
    --input_dir test/input \
    --output_dir test/output
```

#### Inference Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--config` | ✅ | Path to YAML config file |
| `--weights` | ✅ | Path to model weights file |
| `--input_dir` | ✅ | Input image directory or single image |
| `--output_dir` | ❌ | Output directory, default `./results` |
| `--tile` | ❌ | Tile size for large image inference |
| `--tile_overlap` | ❌ | Tile overlap, default 32 |

---

### Acknowledgement

This code is based on [Restormer](https://github.com/swz30/Restormer).

```bibtex
@inproceedings{Zamir2021Restormer,
    title={Restormer: Efficient Transformer for High-Resolution Image Restoration}, 
    author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat 
            and Fahad Shahbaz Khan and Ming-Hsuan Yang},
    booktitle={CVPR},
    year={2022}
}
```
