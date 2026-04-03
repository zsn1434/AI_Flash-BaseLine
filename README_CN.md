# AIFlash - Baseline

[英文说明 (English)](README.md)

---

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


### 📥 数据集下载

当前数据集已全部公开（包含各阶段与 GT），推荐从 Hugging Face 获取；原比赛页面保留以供参考：

- 公共数据集：[https://huggingface.co/datasets/zsn1434/AI_Flash_Portrait](https://huggingface.co/datasets/zsn1434/AI_Flash_Portrait/tree/main)
- 比赛页面：[https://www.codabench.org/competitions/12885/](https://www.codabench.org/competitions/12885/)

| 阶段 | 数据集 | 状态 | 说明 |
|------|--------|------|------|
| 第一阶段 | 训练集 | ✅ 已开放 | 训练图像（输入 + GT 配对 + 人像掩码） |
| 第二阶段 | 测试集 A | ✅ 已开放（公开且含 GT） | 测试图像与 GT 已在 Hugging Face 公开 |
| 第三阶段 | 测试集 B | ✅ 已开放（公开且含 GT） | 测试图像与 GT 已在 Hugging Face 公开 |


#### 目录结构

下载后，请按以下结构组织数据：

```
AI_Flash-BaseLine/
├── datasets/
│   ├── train/
│   │   ├── input/          # 训练输入图像（暗光）
│   │   │   ├── 0001.jpg
│   │   │   ├── 0002.jpg
│   │   │   └── ...
│   │   ├── gt/             # 训练真值图像
│   │   │   ├── 0001.jpg
│   │   │   ├── 0002.jpg
│   │   │   └── ...
│   │   └── mask_personmask/    # 人像分割掩码（如使用）
│   │       ├── 0001.jpg
│   │       ├── 0002.jpg
│   │       └── ...
│   └── test/
│       ├── input/          # 测试输入图像
│       │   ├── 0001.jpg
│       │   ├── 0002.jpg
│       │   └── ...
│       └── mask_personmask/    # 人像分割掩码（如使用）
│           ├── 0001.jpg
│           ├── 0002.jpg
│           └── ...
├── configs/
├── basicsr/
└── ...
```

---


### 环境安装

#### 1. 创建 Conda 环境

```bash
conda create -n aiflash python=3.10 -y
conda activate aiflash
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
cd AI_Flash-BaseLine
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
    --weights pth/net_g_xxx.pth \
    --input_dir test/input \
    --output_dir test/output
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
```bibtex
@inproceedings{Zamir2021Restormer,
    title={Restormer: Efficient Transformer for High-Resolution Image Restoration}, 
    author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat 
            and Fahad Shahbaz Khan and Ming-Hsuan Yang},
    booktitle={CVPR},
    year={2022}
}
```
