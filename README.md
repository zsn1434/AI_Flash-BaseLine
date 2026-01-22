# AIFlash - Baseline

[English](#english) | [中文](#中文)

---

<a name="中文"></a>

## 🇨🇳 中文

基于 [Restormer](https://github.com/swz30/Restormer) 改进的AI闪光人像 Baseline 代码。

### 🎯 挑战动机

尽管深度学习在图像恢复领域取得了显著进展，但在真实世界的暗光人像场景中，现有模型仍难以兼顾噪声去除、细节保留、光影与色彩还原。**AI 闪光人像**旨在生成自然、干净、真实的人像，同时保持背景观感舒适、色彩准确。

### 🏆 挑战目标

本次挑战赛旨在为工业与学术参与者提供平台，在真实世界成像场景中测试和评估算法与模型。挑战目标包括：

- 建立真实场景暗光人像恢复评测基准，覆盖客观与主观评价；
- 推动高鲁棒性算法的研发与落地应用；
- 鼓励人像与场景整体质量兼优的解决方案。

---

### ⚠️ 数据说明

| 数据 | 分辨率 |
|------|--------|
| 输入图像 (Input) | **1K** |
| 真值图像 (GT) | **1K** |
| 提交结果 | **1K** |

> **注意**：当前 Baseline 训练时将图像 resize 到 **768** 进行训练，选手可根据自己的显存情况调整训练分辨率。最终提交的推理结果必须为 **1K 分辨率**。

---

### 环境安装

#### 1. 创建 Conda 环境

```bash
conda create -n restormer python=3.10 -y
conda activate restormer
```

#### 2. 安装 PyTorch (CUDA 12.1)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> 其他 CUDA 版本请参考 [PyTorch 官网](https://pytorch.org/get-started/locally/)

#### 3. 安装其他依赖

```bash
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb "numpy<2" pyyaml requests scipy tensorboard yapf lpips
```

#### 4. 安装 BasicSR

```bash
cd Restormer-main
pip install -e .
```

---

### 训练

```bash
python basicsr/train.py -opt configs/flash.yaml
```

---

### 推理

```bash
python inference.py \
    --config configs/flash.yaml \
    --weights experiments/RestormerV1_good_train/models/net_g_95000.pth \
    --input_dir ./test/input \
    --output_dir ./test/output
```

#### 推理参数说明

| 参数 | 必填 | 说明 |
|------|------|------|
| `--config` | ✅ | YAML 配置文件路径 |
| `--weights` | ✅ | 模型权重文件路径 |
| `--input_dir` | ✅ | 输入图像目录或单张图片 |
| `--output_dir` | ❌ | 输出目录，默认 `./results` |
| `--tile` | ❌ | 分块大小，用于大图推理 |
| `--tile_overlap` | ❌ | 分块重叠，默认 32 |

---

### 致谢

本代码基于 [Restormer](https://github.com/swz30/Restormer) 修改。

---

<a name="english"></a>
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

### Environment Setup

#### 1. Create Conda Environment

```bash
conda create -n restormer python=3.10 -y
conda activate restormer
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
cd Restormer-main
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
    --weights experiments/RestormerV1_good_train/models/net_g_95000.pth \
    --input_dir ./test/input \
    --output_dir ./test/output
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
