import os
import glob
import math
import json
import csv
import yaml
import cv2
import re
import numpy as np
from sys import argv

try:
    import torch
except Exception:
    torch = None

try:
    import pyiqa
except Exception:
    pyiqa = None


-----------------------------
Debug switch
-----------------------------
DEBUG = False  


def dprint(msg: str):
    if DEBUG:
        print(msg)


def ls(pattern):
    return sorted(glob.glob(pattern))


ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

允许文件名任意位置出现 3 位数字 ID：001_flash.png / flash_001.png / x001y.png 都能抽出来
ID_RE = re.compile(r"(\d{3})")


def collect_images(dir_path):
    """
    Recursively collect images under dir_path.
    Key by extracted 3-digit ID (e.g., "001").
    Ignore non-image files, and files with 'mask' in stem as extra safety.
    """
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    images = {}

    dprint(f"[DEBUG] 正在读取文件夹: {dir_path}")
    try:
        dprint(f"[DEBUG] 文件夹内所有文件(仅当前层): {os.listdir(dir_path)}")
    except Exception as e:
        dprint(f"[DEBUG] os.listdir 失败: {e}")

    for root, dirs, files in os.walk(dir_path):
        # 跳过包含 "mask" 的子目录，防止把 mask 文件当作图片扫描
        dirs[:] = [d for d in dirs if "mask" not in d.lower()]

        for name in files:
            ext = os.path.splitext(name)[1].lower()
            file_path = os.path.join(root, name)

            dprint(f"[DEBUG] 检测文件: {name} | 后缀: {ext} | 完整路径: {file_path}")

            if ext not in ALLOWED_EXTS:
                dprint(f"[DEBUG] ? 过滤非图片文件: {name}")
                continue

            stem = os.path.splitext(name)[0]
            if "mask" in stem.lower():
                dprint(f"[DEBUG] ? 过滤掩码相关文件: {name}")
                continue

            match = ID_RE.search(stem)
            if match is None:
                dprint(f"[DEBUG] ? 无3位数字ID，过滤文件: {name}")
                continue

            img_id = match.group(1)
            dprint(f"[DEBUG] ? 有效图片: {name} | 提取的图片ID: {img_id}")

            if img_id in images:
                raise ValueError(
                    f"Duplicate image ID '{img_id}' found:\n"
                    f"  - {images[img_id]}\n"
                    f"  - {file_path}"
                )

            images[img_id] = file_path

    dprint(f"[DEBUG] 文件夹 {dir_path} 最终有效图片数量: {len(images)}")
    if DEBUG:
        dprint(f"[DEBUG] 有效图片ID和路径: {images}")

    if len(images) == 0:
        raise ValueError(
            f"No valid images found in {dir_path}. "
            f"Expected files like 001.png / 001_flash.png / flash_001.png ..."
        )

    return images


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def load_mask(path, target_hw):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.dtype != np.float32:
        mask = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0
    if mask.shape[:2] != target_hw:
        mask = cv2.resize(mask, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0.5).astype(np.float32)


def mse_on_mask(a, b, mask=None):
    diff = (a - b) ** 2
    if mask is None:
        return float(diff.mean())
    if mask.sum() == 0:
        return None
    if diff.ndim == 3:
        diff = diff.mean(axis=2)
    return float((diff * mask).sum() / mask.sum())


def psnr(a, b, mask=None):
    err = mse_on_mask(a, b, mask)
    if err is None:
        return None
    if err == 0:
        return 100.0
    return 20.0 * math.log10(1.0 / math.sqrt(err))


def to_torch_image(img, device):
    t = torch.from_numpy(img).to(device)
    t = t.permute(2, 0, 1).unsqueeze(0)  # NCHW
    return t


def to_torch_mask(mask, device, hw):
    if mask is None:
        return None
    if mask.shape != hw:
        mask = cv2.resize(mask, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    m = torch.from_numpy(mask).to(device)
    m = m.unsqueeze(0).unsqueeze(0)  # N1HW
    return m


def psnr_torch(sr, hr, mask=None):
    diff = (sr - hr) ** 2
    if mask is None:
        err = diff.mean()
    else:
        per_pixel = diff.mean(dim=1, keepdim=True)
        denom = mask.sum()
        if denom == 0:
            return None
        err = (per_pixel * mask).sum() / denom
    err = float(err.item())
    if err == 0:
        return 100.0
    return 20.0 * math.log10(1.0 / math.sqrt(err))


def ssim_torch(sr, hr, mask=None):
    # SSIM on luminance, gaussian kernel 5x5
    y1 = 0.299 * sr[:, 0:1] + 0.587 * sr[:, 1:2] + 0.114 * sr[:, 2:3]
    y2 = 0.299 * hr[:, 0:1] + 0.587 * hr[:, 1:2] + 0.114 * hr[:, 2:3]
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    kernel = torch.tensor(
        [
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ],
        dtype=sr.dtype,
        device=sr.device,
    )
    kernel = (kernel / kernel.sum()).view(1, 1, 5, 5)
    padding = 2

    mu1 = torch.nn.functional.conv2d(y1, kernel, padding=padding)
    mu2 = torch.nn.functional.conv2d(y2, kernel, padding=padding)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(y1 * y1, kernel, padding=padding) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(y2 * y2, kernel, padding=padding) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(y1 * y2, kernel, padding=padding) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    if mask is None:
        return float(ssim_map.mean().item())
    denom = mask.sum()
    if denom == 0:
        return None
    return float((ssim_map * mask).sum().item() / float(denom.item()))


def ssim_np(a, b, mask=None):
    y1 = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    y2 = 0.299 * b[:, :, 0] + 0.587 * b[:, :, 1] + 0.114 * b[:, :, 2]
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu1 = cv2.GaussianBlur(y1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(y2, (11, 11), 1.5)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(y1 * y1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(y2 * y2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(y1 * y2, (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    if mask is None:
        return float(ssim_map.mean())
    if mask.sum() == 0:
        return None
    return float((ssim_map * mask).sum() / mask.sum())


def deltae_cie76(a, b, mask=None):
    lab1 = cv2.cvtColor((a * 255.0).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab2 = cv2.cvtColor((b * 255.0).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    diff = lab1 - lab2
    de = np.sqrt(np.sum(diff * diff, axis=2))
    if mask is None:
        return float(de.mean())
    if mask.sum() == 0:
        return None
    return float((de * mask).sum() / mask.sum())


def normalize_value(value, norm):
    """
    norm:
      type: minmax | invert_minmax
      min: ...
      max: ...
    """
    if norm is None:
        return value
    min_v = float(norm.get("min", 0.0))
    max_v = float(norm.get("max", 1.0))
    if max_v == min_v:
        return 0.0
    scaled = (float(value) - min_v) / (max_v - min_v)
    if norm.get("type") == "invert_minmax":
        scaled = 1.0 - scaled
    return max(0.0, min(1.0, scaled))


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_mask(mask_dir, sample_id, patterns):
    for pat in patterns:
        candidate = os.path.join(mask_dir, pat.format(id=sample_id))
        if DEBUG:
            # 只在找不到的时候打印，或者针对特定ID打印，防止刷屏
            if sample_id == "001":
                print(f"[DEBUG-FindMask] Checking: {candidate} | Exists: {os.path.exists(candidate)}")
        if os.path.exists(candidate):
            return candidate
    return None


def crop_by_mask(img, mask, pad=8):
    """
    img: HWC float32 [0,1]
    mask: HW float32 {0,1}
    """
    if mask is None:
        return img
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0 or len(ys) == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(img.shape[0] - 1, y1 + pad)
    x1 = min(img.shape[1] - 1, x1 + pad)
    return img[y0 : y1 + 1, x0 : x1 + 1, :]


-----------------------------
pyiqa metric cache 【仅保留LPIPS，已删除DISTS】
-----------------------------
_PYIQA_LPIPS = None


def get_pyiqa_metric(name: str, device):
    if pyiqa is None:
        raise RuntimeError(f"Metric '{name}' requested but 'pyiqa' is not available.")
    metric = pyiqa.create_metric(name, device=device)
    metric.eval()
    return metric


def get_lpips_metric(device):
    global _PYIQA_LPIPS
    if _PYIQA_LPIPS is None:
        _PYIQA_LPIPS = get_pyiqa_metric("lpips", device)
    return _PYIQA_LPIPS


def get_metric_cfg(metrics_cfg, name):
    for m in metrics_cfg:
        if m.get("name") == name:
            return m
    return None


def compute_metrics(sr_list, hr_list, config, input_dir):
    """
    Compute per-metric means according to config.metrics 【已删除DISTS计算分支】

    Returns:
        metric_means: dict, 各指标的平均值
        per_image_results: list of dict, 每张图片的详细评分结果
    """
    metrics_cfg = config.get("metrics", [])
    # mask_dir = os.path.join(input_dir, config.get("mask_dir", "mask"))
    # [FIX] 强制使用 ref/mask_personmask 结构，与 Codabench 上的解压结构保持一致
    # 因为 config.get("mask_dir") 可能是 "ref/mask_personmask"，拼接后变成 input_dir/ref/mask_personmask
    # 但如果 input_dir 本身包含了 ref (例如本地测试时)，可能会导致路径重复

    # 优先尝试从 config 读取，但要处理潜在的路径问题
    cfg_mask_dir = config.get("mask_dir", "mask")
    mask_dir = os.path.join(input_dir, cfg_mask_dir)

    # 二次确认：如果拼出来的路径不存在，尝试在 input_dir/ref 下找
    if not os.path.exists(mask_dir):
        alt_mask_dir = os.path.join(input_dir, "ref", "mask_personmask")
        dprint(f"[DEBUG] mask_dir ({mask_dir}) 不存在，尝试: {alt_mask_dir}")
        if os.path.exists(alt_mask_dir):
            mask_dir = alt_mask_dir

    mask_patterns = config.get("mask_patterns", {})

    dprint(f"[DEBUG] compute_metrics mask_dir: {mask_dir}")
    dprint(f"[DEBUG] compute_metrics mask_patterns: {mask_patterns}")

    use_torch = bool(config.get("use_torch", True)) and (torch is not None)
    device = torch.device("cuda") if use_torch and torch.cuda.is_available() else (torch.device("cpu") if use_torch else None)
    dprint(f"[DEBUG] 使用计算设备: {device}")

    per_metric_values = {m["name"]: [] for m in metrics_cfg}
    per_image_results = []  # 存储每张图片的详细评分

    for sr_path, hr_path in zip(sr_list, hr_list):
        # 注意：sample_id 用于找 mask，必须是纯 id（"001"）
        # 这里从文件名里同样抽取 3 位数字，保证与 collect_images 的 key 一致
        stem = os.path.splitext(os.path.basename(sr_path))[0]
        m_id = ID_RE.search(stem)
        sample_id = m_id.group(1) if m_id else stem

        sr = load_image(sr_path)
        hr = load_image(hr_path)

        dprint(f"[DEBUG] 正在计算图片: {sample_id} | sr: {os.path.basename(sr_path)} | 尺寸: {sr.shape}")

        if sr.shape != hr.shape:
            raise ValueError(f"Shape mismatch for {sample_id}: {sr.shape} vs {hr.shape}")

        # 当前图片的评分记录
        image_result = {
            "image_id": sample_id,
            "sr_file": os.path.basename(sr_path),
            "hr_file": os.path.basename(hr_path),
        }

        # load masks
        masks = {}
        for mask_name, patterns in mask_patterns.items():
            mask_path = find_mask(mask_dir, sample_id, patterns)
            if mask_path:
                dprint(f"[DEBUG] 成功加载 Mask '{mask_name}': {mask_path}")
                masks[mask_name] = load_mask(mask_path, sr.shape[:2])
            else:
                dprint(f"[WARNING] 未找到 Mask '{mask_name}' (ID: {sample_id})，将使用全局计算！")
                masks[mask_name] = None

        # torch tensors
        if device is not None:
            sr_t = to_torch_image(sr, device)
            hr_t = to_torch_image(hr, device)
            masks_t = {name: to_torch_mask(m, device, sr.shape[:2]) for name, m in masks.items()}
        else:
            sr_t = hr_t = None
            masks_t = {}

        for m in metrics_cfg:
            metric_name = m["name"]
            metric_type = m.get("type")
            mask_name = m.get("mask")

            # 获取当前 metric 需要的基础 mask
            cur_mask_np = masks.get(mask_name) if mask_name else None
            cur_mask_t = masks_t.get(mask_name) if (device is not None and mask_name) else None

            # 打印 Mask 使用情况
            if mask_name:
                status = "Loaded" if cur_mask_np is not None else "MISSING (Using Global)"
                dprint(f"[DEBUG] Metric '{metric_name}' using mask '{mask_name}': {status}")
            else:
                dprint(f"[DEBUG] Metric '{metric_name}' is Global (No mask specified)")

            # 处理反转逻辑 (invert_mask: true)
            if m.get("invert_mask", False):
                if cur_mask_np is not None:
                    cur_mask_np = 1.0 - cur_mask_np
                if cur_mask_t is not None:
                    cur_mask_t = 1.0 - cur_mask_t

            if metric_type == "psnr":
                value = psnr_torch(sr_t, hr_t, cur_mask_t) if device is not None else psnr(sr, hr, cur_mask_np)

            elif metric_type == "ssim":
                value = ssim_torch(sr_t, hr_t, cur_mask_t) if device is not None else ssim_np(sr, hr, cur_mask_np)

            elif metric_type == "deltae":
                value = deltae_cie76(sr, hr, cur_mask_np)

            elif metric_type == "lpips":
                if device is None:
                    raise RuntimeError("LPIPS requires torch. Set use_torch=true and ensure torch is installed.")
                pad = int(m.get("pad", 8))
                # LPIPS 裁剪使用 numpy mask
                crop_sr = crop_by_mask(sr, cur_mask_np, pad=pad)
                crop_hr = crop_by_mask(hr, cur_mask_np, pad=pad)
                if crop_sr is None or crop_hr is None:
                    value = None
                else:
                    lpips_metric = get_lpips_metric(device)
                    sr_tt = to_torch_image(crop_sr, device)  # [0,1] NCHW
                    hr_tt = to_torch_image(crop_hr, device)
                    with torch.no_grad():
                        dist = lpips_metric(sr_tt, hr_tt)
                    value = float(dist.mean().item())

            else:
                raise ValueError(f"Unknown metric type: {metric_type}")

            if value is not None:
                per_metric_values[metric_name].append(value)
                image_result[metric_name] = value
                dprint(f"[DEBUG] 指标 {metric_name} 得分: {value:.6f}")
            else:
                image_result[metric_name] = None

        per_image_results.append(image_result)

    metric_means = {}
    for m in metrics_cfg:
        name = m["name"]
        values = per_metric_values.get(name, [])
        if len(values) == 0:
            continue
        metric_means[name] = float(np.mean(values))

    if len(metric_means) == 0:
        raise ValueError("No valid metrics were computed. Check images/masks/config.")
    return metric_means, per_image_results


def compute_adaptive_score(lpips_n, deltae_n, global_score_n):
    """
    动态自适应权重评分

    原理：
    - 基础权重：LPIPS 30%, DeltaE 30%, GlobalScore 40%
    - 动态调整：表现越差的指标，权重会被放大（指数级增长）
    - 缺陷度 defect = 1.0 - score，缺陷越大，权重越高
    - 公式：new_weight = base_weight * exp(defect * sensitivity)

    Args:
        lpips_n: 归一化后的 LPIPS 分数 (0~1, 越大越好)
        deltae_n: 归一化后的 DeltaE 分数 (0~1, 越大越好)
        global_score_n: GlobalScore (0~1)

    Returns:
        (score_0_to_1, final_weights_dict)
    """
    # 1. 基础权重
    weights = {'lpips': 0.3, 'deltae': 0.3, 'global': 0.4}
    scores = {'lpips': lpips_n, 'deltae': deltae_n, 'global': global_score_n}

    # 2. 动态调整：越差的项，权重越高
    # 敏感度因子：越大惩罚越重，2.0 是一个平衡值
    sensitivity = 2.0

    adj_weights = {}
    sum_w = 0
    for k, w in weights.items():
        defect = 1.0 - scores[k]
        # 如果得分很低 (defect大)，权重会变大
        aw = w * np.exp(defect * sensitivity)
        adj_weights[k] = aw
        sum_w += aw

    # 3. 归一化权重
    final_weights = {k: w / sum_w for k, w in adj_weights.items()}

    # 4. 计算总分 (0~1)
    total = 0
    for k in scores:
        total += scores[k] * final_weights[k]

    return total, final_weights


def compute_total_score(metric_means, config):
    """
    计算整体 TotalScore（使用动态自适应权重）

    公式：TotalScore = 30 + 70 * AdaptiveScore
    其中 AdaptiveScore 基于动态权重，表现差的指标权重更高
    """
    metrics_cfg = config.get("metrics", [])

    def must(name):
        if name not in metric_means:
            raise ValueError(
                f"Required metric '{name}' missing. "
                f"Computed metrics: {sorted(metric_means.keys())}. "
                f"Check config.metrics and whether masks are present/non-empty."
            )
        return metric_means[name]

    def get_optional(name, default=0.0):
        """获取可选指标，不存在则返回默认值"""
        return metric_means.get(name, default)

    # 核心指标
    psnr_bg = must("PSNR_bg")
    ssim_global = must("SSIM_global")
    deltae_person = must("DeltaE_person")
    lpips_person = get_optional("LPIPS_person", 0.2)
    has_lpips = "LPIPS_person" in metric_means

    # 各项指标归一化
    psnr_bg_cfg = get_metric_cfg(metrics_cfg, "PSNR_bg")
    psnr_bg_n = normalize_value(psnr_bg, psnr_bg_cfg.get("normalize") if psnr_bg_cfg else None)

    ssim_global_cfg = get_metric_cfg(metrics_cfg, "SSIM_global")
    ssim_global_n = normalize_value(ssim_global, ssim_global_cfg.get("normalize") if ssim_global_cfg else None)

    deltae_cfg = get_metric_cfg(metrics_cfg, "DeltaE_person")
    lpips_cfg = get_metric_cfg(metrics_cfg, "LPIPS_person")

    deltae_n = normalize_value(deltae_person, deltae_cfg.get("normalize") if deltae_cfg else None)
    lpips_n = normalize_value(lpips_person, lpips_cfg.get("normalize") if lpips_cfg else None)

    # 非线性映射：S_new = S_old ^ 1.2 (拉开高分段差距)
    lpips_n_mapped = lpips_n ** 1.2
    deltae_n_mapped = deltae_n ** 1.2
    ssim_global_n_mapped = ssim_global_n ** 1.2
    psnr_bg_n_mapped = psnr_bg_n ** 1.2

    GlobalScore_mapped = 0.5 * psnr_bg_n_mapped + 0.5 * ssim_global_n_mapped

    # 使用动态自适应权重
    adaptive_score, final_weights = compute_adaptive_score(lpips_n_mapped, deltae_n_mapped, GlobalScore_mapped)

    # 混合策略：TotalScore = 30 + 70 * AdaptiveScore
    # - 最差情况（各项全0）：30分
    # - 最好情况（各项全1）：100分
    TotalScore = 30.0 + 70.0 * adaptive_score

    # 调试用归一化值
    debug_norm = {
        "Norm(PSNR_bg)": psnr_bg_n,
        "SSIM_global_used": ssim_global_n,
        "Norm(DeltaE_person)": deltae_n,
        "Norm(LPIPS_person)": lpips_n
    }
    if not has_lpips:
        debug_norm["LPIPS_person_note"] = "LPIPS disabled, using default value"
    return TotalScore, GlobalScore_mapped, debug_norm


Default I/O directories:
智能判断：如果 /app 存在，说明是在 Codabench 容器内，使用绝对路径
if os.path.exists("/app"):
    root_dir = "/app"
else:
    root_dir = "./"  # 本地测试

default_input_dir = os.path.join(root_dir, "input")
default_output_dir = os.path.join(root_dir, "output")

scoring_version = 3.0
default_config_path = os.path.join(os.path.dirname(file), "metrics_config.yaml")


def parse_args():
    """
    解析命令行参数，支持两种模式：
    1. 本地测试模式: python evaluate.py --ref <ref_dir> --res <res_dir> --output <output_dir>
    2. Codabench 模式: python evaluate.py <input_dir> <output_dir>
    """
    import argparse

    parser = argparse.ArgumentParser(description="RAIM Image Restoration Scoring Program")

    # 本地测试模式参数
    parser.add_argument("--ref", type=str, default=None,
                        help="本地测试: GT图片目录路径")
    parser.add_argument("--res", type=str, default=None,
                        help="本地测试: 待评估结果图片目录路径")
    parser.add_argument("--mask", type=str, default=None,
                        help="本地测试: Mask目录路径 (包含 mask_personmask 等子文件夹)")
    parser.add_argument("--output", "-o", type=str, default="./scoring_output",
                        help="输出目录路径 (默认: ./scoring_output)")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="配置文件路径 (默认使用同目录下的 metrics_config.yaml)")
    parser.add_argument("--debug", action="store_true", default=True,
                        help="开启调试模式 (默认开启)")
    parser.add_argument("--no-debug", dest="debug", action="store_false",
                        help="关闭调试模式")

    # Codabench 兼容: 位置参数
    parser.add_argument("input_dir", nargs="?", default=None,
                        help="Codabench模式: input目录")
    parser.add_argument("output_dir", nargs="?", default=None,
                        help="Codabench模式: output目录")

    return parser.parse_args()


def run_scoring(ref_dir, res_dir, mask_dir, output_dir, config_path, is_local_mode=False):
    """
    执行评分核心逻辑

    Args:
        ref_dir: GT图片目录
        res_dir: 待评估结果图片目录
        mask_dir: Mask目录 (包含 mask_personmask 等子文件夹)
        output_dir: 输出目录
        config_path: 配置文件路径
        is_local_mode: 是否为本地测试模式
    """
    dprint(f"[DEBUG] ref文件夹路径: {ref_dir}")
    dprint(f"[DEBUG] res文件夹路径: {res_dir}")
    dprint(f"[DEBUG] mask文件夹路径: {mask_dir}")
    dprint(f"[DEBUG] output文件夹路径: {output_dir}")
    dprint(f"[DEBUG] config文件路径: {config_path}")
    dprint(f"[DEBUG] 本地测试模式: {is_local_mode}")

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Debugging Mask Path ---
    mask_personmask_dir = os.path.join(mask_dir, "mask_personmask") if mask_dir else None
    if mask_personmask_dir:
        dprint(f"[DEBUG] 预期的 Mask 文件夹路径: {mask_personmask_dir}")
        if os.path.exists(mask_personmask_dir):
            try:
                dprint(f"[DEBUG] Mask 文件夹存在，内容: {os.listdir(mask_personmask_dir)[:10]}")
            except Exception as e:
                dprint(f"[WARNING] 无法读取 Mask 文件夹: {e}")
        else:
            dprint(f"[WARNING] Mask 文件夹不存在!")
            if mask_dir and os.path.exists(mask_dir):
                dprint(f"[DEBUG] {mask_dir} 内容: {os.listdir(mask_dir)}")
    # ---------------------------

    # 读取图片（鲁棒ID提取 + debug）
    ref_map = collect_images(ref_dir)
    res_map = collect_images(res_dir)

    dprint(f"[DEBUG] ref的有效图片ID: {sorted(ref_map.keys())}")
    dprint(f"[DEBUG] res的有效图片ID: {sorted(res_map.keys())}")

    missing = sorted(set(ref_map.keys()) - set(res_map.keys()))
    extra = sorted(set(res_map.keys()) - set(ref_map.keys()))
    if missing or extra:
        msg = []
        if missing:
            msg.append(f"Missing predictions: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if extra:
            msg.append(f"Extra predictions: {extra[:5]}{'...' if len(extra) > 5 else ''}")
        raise ValueError("Mismatch between ref and res.\n" + "\n".join(msg))

    keys = sorted(ref_map.keys())
    hr_list = [ref_map[k] for k in keys]
    sr_list = [res_map[k] for k in keys]
    dprint(f"[DEBUG] picture number: {len(keys)} | 匹配的ID: {keys}")

    # 加载配置
    config = load_config(config_path)

    # 本地模式下覆盖 mask_dir
    if is_local_mode and mask_dir:
        config["mask_dir_override"] = mask_dir

    # 1) compute metric means (现在同时返回每张图片的详细评分)
    # 在本地模式下，需要传入 mask_dir 而不是 input_dir
    if is_local_mode:
        metric_means, per_image_results = compute_metrics_local(sr_list, hr_list, config, mask_dir)
    else:
        input_dir = os.path.dirname(os.path.dirname(ref_dir))  # 推断 input_dir
        metric_means, per_image_results = compute_metrics(sr_list, hr_list, config, input_dir)

    # 2) compute TotalScore by modified formula
    total_score, global_score, debug_norm = compute_total_score(metric_means, config)

    # 输出结果
    score_file_path = os.path.join(output_dir, "scores.txt")
    with open(score_file_path, "w", encoding="utf-8") as score_file:
        score_file.write("Score: %0.6f\n" % float(total_score))
        score_file.write("GlobalScore: %0.6f\n" % float(global_score))

        # raw metric means
        for name, value in metric_means.items():
            score_file.write(f"{name}: {value:.6f}\n")

        # normalized components
        for name, value in debug_norm.items():
            if isinstance(value, (int, float)):
                score_file.write(f"{name}: {float(value):.6f}\n")
            else:
                score_file.write(f"{name}: {value}\n")

        score_file.write("Duration: 0\n")

    # 同时输出 scores.json (兼容 Codabench)
    scores_dict = {
        "Score": float(total_score),
        "GlobalScore": float(global_score),
    }
    for name, value in metric_means.items():
        scores_dict[name] = float(value)
    for name, value in debug_norm.items():
        if isinstance(value, (int, float)):
            scores_dict[name] = float(value)

    json_file_path = os.path.join(output_dir, "scores.json")
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(scores_dict, json_file, indent=2, ensure_ascii=False)

    # ========== 输出每张图片的评分到 CSV ==========
    csv_file_path = os.path.join(output_dir, "per_image_scores.csv")
    if per_image_results:
        # 获取所有的指标名称作为 CSV 列头
        metric_names = [m["name"] for m in config.get("metrics", [])]
        fieldnames = ["image_id", "sr_file", "hr_file"] + metric_names

        with open(csv_file_path, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for row in per_image_results:
                # 格式化数值为小数点后6位
                formatted_row = {}
                for k, v in row.items():
                    if isinstance(v, float):
                        formatted_row[k] = f"{v:.6f}"
                    elif v is None:
                        formatted_row[k] = "N/A"
                    else:
                        formatted_row[k] = v
                writer.writerow(formatted_row)

        print(f"  CSV文件: {csv_file_path}")

    print("=" * 60)
    print(f"[SUCCESS] 评分完成!")
    print(f"  Total Score: {total_score:.4f}")
    print(f"  Global Score: {global_score:.4f}")
    print(f"  输出文件: {score_file_path}")
    print(f"  JSON文件: {json_file_path}")
    print(f"  CSV文件: {csv_file_path}")
    print("=" * 60)

    # 打印详细指标
    print("\n[指标详情]")
    for name, value in metric_means.items():
        print(f"  {name}: {value:.6f}")
    print("\n[归一化指标]")
    for name, value in debug_norm.items():
        if isinstance(value, (int, float)):
            print(f"  {name}: {value:.6f}")
        else:
            print(f"  {name}: {value}")

    print(f"\n[每张图片评分] 共 {len(per_image_results)} 张图片，详情见 {csv_file_path}")

    return total_score, global_score, metric_means, per_image_results


def compute_metrics_local(sr_list, hr_list, config, mask_dir):
    """
    本地测试专用的指标计算函数
    与 compute_metrics 类似，但 mask_dir 直接传入

    Returns:
        metric_means: dict, 各指标的平均值
        per_image_results: list of dict, 每张图片的详细评分结果
    """
    metrics_cfg = config.get("metrics", [])
    mask_patterns = config.get("mask_patterns", {})

    dprint(f"[DEBUG] compute_metrics_local mask_dir: {mask_dir}")
    dprint(f"[DEBUG] compute_metrics_local mask_patterns: {mask_patterns}")

    use_torch = bool(config.get("use_torch", True)) and (torch is not None)
    device = torch.device("cuda") if use_torch and torch.cuda.is_available() else (torch.device("cpu") if use_torch else None)
    dprint(f"[DEBUG] 使用计算设备: {device}")

    per_metric_values = {m["name"]: [] for m in metrics_cfg}
    per_image_results = []  # 存储每张图片的详细评分

    for sr_path, hr_path in zip(sr_list, hr_list):
        stem = os.path.splitext(os.path.basename(sr_path))[0]
        m_id = ID_RE.search(stem)
        sample_id = m_id.group(1) if m_id else stem

        sr = load_image(sr_path)
        hr = load_image(hr_path)

        dprint(f"[DEBUG] 正在计算图片: {sample_id} | sr: {os.path.basename(sr_path)} | 尺寸: {sr.shape}")

        if sr.shape != hr.shape:
            raise ValueError(f"Shape mismatch for {sample_id}: {sr.shape} vs {hr.shape}")

        # 当前图片的评分记录
        image_result = {
            "image_id": sample_id,
            "sr_file": os.path.basename(sr_path),
            "hr_file": os.path.basename(hr_path),
        }

        # load masks
        masks = {}
        for mask_name, patterns in mask_patterns.items():
            mask_path = find_mask(mask_dir, sample_id, patterns) if mask_dir else None
            if mask_path:
                dprint(f"[DEBUG] 成功加载 Mask '{mask_name}': {mask_path}")
                masks[mask_name] = load_mask(mask_path, sr.shape[:2])
            else:
                dprint(f"[WARNING] 未找到 Mask '{mask_name}' (ID: {sample_id})，将使用全局计算！")
                masks[mask_name] = None

        # torch tensors
        if device is not None:
            sr_t = to_torch_image(sr, device)
            hr_t = to_torch_image(hr, device)
            masks_t = {name: to_torch_mask(m, device, sr.shape[:2]) for name, m in masks.items()}
        else:
            sr_t = hr_t = None
            masks_t = {}

        for m in metrics_cfg:
            metric_name = m["name"]
            metric_type = m.get("type")
            mask_name = m.get("mask")

            cur_mask_np = masks.get(mask_name) if mask_name else None
            cur_mask_t = masks_t.get(mask_name) if (device is not None and mask_name) else None

            if mask_name:
                status = "Loaded" if cur_mask_np is not None else "MISSING (Using Global)"
                dprint(f"[DEBUG] Metric '{metric_name}' using mask '{mask_name}': {status}")
            else:
                dprint(f"[DEBUG] Metric '{metric_name}' is Global (No mask specified)")

            # 处理反转逻辑
            if m.get("invert_mask", False):
                if cur_mask_np is not None:
                    cur_mask_np = 1.0 - cur_mask_np
                if cur_mask_t is not None:
                    cur_mask_t = 1.0 - cur_mask_t

            if metric_type == "psnr":
                value = psnr_torch(sr_t, hr_t, cur_mask_t) if device is not None else psnr(sr, hr, cur_mask_np)
            elif metric_type == "ssim":
                value = ssim_torch(sr_t, hr_t, cur_mask_t) if device is not None else ssim_np(sr, hr, cur_mask_np)
            elif metric_type == "deltae":
                value = deltae_cie76(sr, hr, cur_mask_np)
            elif metric_type == "lpips":
                if device is None:
                    raise RuntimeError("LPIPS requires torch.")
                pad = int(m.get("pad", 8))
                crop_sr = crop_by_mask(sr, cur_mask_np, pad=pad)
                crop_hr = crop_by_mask(hr, cur_mask_np, pad=pad)
                if crop_sr is None or crop_hr is None:
                    value = None
                else:
                    lpips_metric = get_lpips_metric(device)
                    sr_tt = to_torch_image(crop_sr, device)
                    hr_tt = to_torch_image(crop_hr, device)
                    with torch.no_grad():
                        dist = lpips_metric(sr_tt, hr_tt)
                    value = float(dist.mean().item())
            else:
                raise ValueError(f"Unknown metric type: {metric_type}")

            if value is not None:
                per_metric_values[metric_name].append(value)
                image_result[metric_name] = value
                dprint(f"[DEBUG] 指标 {metric_name} 得分: {value:.6f}")
            else:
                image_result[metric_name] = None

        per_image_results.append(image_result)

    metric_means = {}
    for m in metrics_cfg:
        name = m["name"]
        values = per_metric_values.get(name, [])
        if len(values) == 0:
            continue
        metric_means[name] = float(np.mean(values))

    if len(metric_means) == 0:
        raise ValueError("No valid metrics were computed.")
    return metric_means, per_image_results


if name == "main":
    import sys
    import traceback

    args = parse_args()
    DEBUG = args.debug  # 更新全局 DEBUG 开关

    # --- Debugging Path Issues ---
    print("=" * 40)
    print("RAIM Scoring Program v3.0")
    print("=" * 40)
    print(f"Current working directory: {os.getcwd()}")
    dprint(f"[DEBUG] 命令行参数: {sys.argv}")

    # 判断运行模式
    is_local_mode = args.ref is not None and args.res is not None

    if is_local_mode:
        # ========== 本地测试模式 ==========
        print("[MODE] 本地测试模式")
        ref_dir = args.ref
        res_dir = args.res
        mask_dir = args.mask if args.mask else os.path.dirname(ref_dir)  # 默认与 ref 同级目录
        output_dir = args.output
        config_path = args.config if args.config else default_config_path

        try:
            run_scoring(ref_dir, res_dir, mask_dir, output_dir, config_path, is_local_mode=True)
        except Exception as e:
            print("=" * 60)
            print("!!! ERROR IN SCORING PROGRAM !!!")
            print(traceback.format_exc())
            print("=" * 60)
            sys.exit(1)

    else:
        # ========== Codabench 模式 ==========
        print("[MODE] Codabench 模式")

        # 确定 input_dir 和 output_dir
        if args.input_dir:
            input_dir = args.input_dir
            output_dir = args.output_dir if args.output_dir else default_output_dir
        else:
            input_dir = default_input_dir
            output_dir = default_output_dir

        dprint(f"[DEBUG] 实际使用输入路径: {input_dir}")
        dprint(f"[DEBUG] 实际使用输出路径: {output_dir}")

        # Debug: 打印目录内容
        print("Contents of /app (if exists):")
        try:
            if os.path.exists("/app"):
                print(os.listdir("/app"))
            else:
                print("  /app directory does not exist.")
        except Exception as e:
            print(f"  Error listing /app: {e}")

        try:
            # 确保输出目录存在
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Codabench 把 reference_data.zip 解压到 /app/input/ref/
            ref_dir = os.path.join(input_dir, "ref", "ref")
            res_dir = os.path.join(input_dir, "res")
            mask_dir = os.path.join(input_dir, "ref")  # mask 在 ref/mask_personmask/

            dprint(f"[DEBUG] ref文件夹路径: {ref_dir}")
            dprint(f"[DEBUG] res文件夹路径: {res_dir}")
            dprint(f"[DEBUG] mask父文件夹路径: {mask_dir}")

            config_path = os.environ.get("RAIM_SCORE_CONFIG", default_config_path)
            config = load_config(config_path)

            # 读取图片
            ref_map = collect_images(ref_dir)
            res_map = collect_images(res_dir)

            dprint(f"[DEBUG] ref的有效图片ID: {sorted(ref_map.keys())}")
            dprint(f"[DEBUG] res的有效图片ID: {sorted(res_map.keys())}")

            missing = sorted(set(ref_map.keys()) - set(res_map.keys()))
            extra = sorted(set(res_map.keys()) - set(ref_map.keys()))
            if missing or extra:
                msg = []
                if missing:
                    msg.append(f"Missing predictions: {missing[:5]}{'...' if len(missing) > 5 else ''}")
                if extra:
                    msg.append(f"Extra predictions: {extra[:5]}{'...' if len(extra) > 5 else ''}")
                raise ValueError("Mismatch between ref and res.\n" + "\n".join(msg))

            keys = sorted(ref_map.keys())
            hr_list = [ref_map[k] for k in keys]
            sr_list = [res_map[k] for k in keys]
            dprint(f"[DEBUG] picture number: {len(keys)} | 匹配的ID: {keys}")

            # 计算指标 (现在同时返回每张图片的详细评分)
            metric_means, per_image_results = compute_metrics(sr_list, hr_list, config, input_dir)
            total_score, global_score, debug_norm = compute_total_score(metric_means, config)

            # 写入结果
            with open(os.path.join(output_dir, "scores.txt"), "w", encoding="utf-8") as score_file:
                score_file.write("Score: %0.6f\n" % float(total_score))
                score_file.write("GlobalScore: %0.6f\n" % float(global_score))
                for name, value in metric_means.items():
                    score_file.write(f"{name}: {value:.6f}\n")
                for name, value in debug_norm.items():
                    if isinstance(value, (int, float)):
                        score_file.write(f"{name}: {float(value):.6f}\n")
                score_file.write("Duration: 0\n")

            # 同时输出 scores.json
            scores_dict = {"Score": float(total_score), "GlobalScore": float(global_score)}
            for name, value in metric_means.items():
                scores_dict[name] = float(value)
            for name, value in debug_norm.items():
                if isinstance(value, (int, float)):
                    scores_dict[name] = float(value)

            with open(os.path.join(output_dir, "scores.json"), "w", encoding="utf-8") as json_file:
                json.dump(scores_dict, json_file, indent=2)

            # ========== 输出每张图片的评分到 CSV ==========
            csv_file_path = os.path.join(output_dir, "per_image_scores.csv")
            if per_image_results:
                metric_names = [m["name"] for m in config.get("metrics", [])]
                fieldnames = ["image_id", "sr_file", "hr_file"] + metric_names

                with open(csv_file_path, "w", encoding="utf-8", newline="") as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction='ignore')
                    writer.writeheader()
                    for row in per_image_results:
                        formatted_row = {}
                        for k, v in row.items():
                            if isinstance(v, float):
                                formatted_row[k] = f"{v:.6f}"
                            elif v is None:
                                formatted_row[k] = "N/A"
                            else:
                                formatted_row[k] = v
                        writer.writerow(formatted_row)

            print(f"[INFO] scores.txt saved to: {os.path.join(output_dir, 'scores.txt')}")
            print(f"[INFO] scores.json saved to: {os.path.join(output_dir, 'scores.json')}")
            print(f"[INFO] per_image_scores.csv saved to: {csv_file_path}")

        except Exception as e:
            # 捕获所有异常，确保输出 scores.txt
            print("=" * 60)
            print("!!! FATAL ERROR IN SCORING PROGRAM !!!")
            print(traceback.format_exc())
            print("=" * 60)

            try:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                with open(os.path.join(output_dir, "scores.txt"), "w", encoding="utf-8") as f:
                    f.write("Score: 0\n")
                    f.write("Error: 1\n")
                    f.write(f"Message: {str(e)}\n")

                with open(os.path.join(output_dir, "scores.json"), "w", encoding="utf-8") as f:
                    json.dump({"Score": 0, "Error": 1, "Message": str(e)}, f)

                with open(os.path.join(output_dir, "error_log.txt"), "w", encoding="utf-8") as f:
                    f.write(traceback.format_exc())

            except Exception as write_err:
                print(f"Failed to write error log: {write_err}")
