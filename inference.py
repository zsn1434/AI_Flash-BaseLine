"""
Restormer Inference Script
从 YAML 配置文件加载模型参数进行推理

使用示例:
    # 基础用法
    python inference.py --config configs/flash.yaml --weights path/to/model.pth --input_dir ./test/input --output_dir ./test/output
    
    # 使用 tile 模式处理大图（显存不足时使用）
    python inference.py --config configs/flash.yaml --weights path/to/model.pth --input_dir ./test/input --output_dir ./test/output --tile 512 --tile_overlap 32
    
    # 处理单张图片
    python inference.py --config configs/flash.yaml --weights path/to/model.pth --input_dir ./test/input/001.png --output_dir ./test/output
"""

import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from skimage import img_as_ubyte


def parse_args():
    parser = argparse.ArgumentParser(description='Restormer Inference')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to YAML config file')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights (.pth file)')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory or single image path')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for restored images')
    parser.add_argument('--tile', type=int, default=None,
                        help='Tile size for processing large images (e.g. 512, 720). None means full resolution.')
    parser.add_argument('--tile_overlap', type=int, default=32,
                        help='Overlap between tiles')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use: cuda or cpu')
    return parser.parse_args()


def load_config(config_path):
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def build_model(config):
    """根据配置构建模型"""
    from basicsr.models.archs.restormer_arch import Restormer
    
    # 从配置文件获取网络参数
    network_config = config['network_g']
    
    # 移除 'type' 字段，只保留模型参数
    model_params = {k: v for k, v in network_config.items() if k != 'type'}
    
    # 构建模型
    model = Restormer(**model_params)
    
    return model


def load_weights(model, weights_path):
    """加载模型权重"""
    print(f'Loading weights from: {weights_path}')
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    # 处理不同格式的权重文件
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 移除可能的 'module.' 前缀（多卡训练产生的）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    print('Weights loaded successfully!')
    return model


def load_img(filepath):
    """加载图像 (BGR -> RGB)"""
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f'Failed to load image: {filepath}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_img(filepath, img):
    """保存图像 (RGB -> BGR)"""
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def get_image_files(input_path):
    """获取输入图像文件列表"""
    extensions = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG', 'bmp', 'BMP']
    
    # 判断是单个文件还是目录
    if os.path.isfile(input_path):
        return [input_path]
    
    # 目录模式：搜索所有图像文件
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(input_path, f'*.{ext}')))
    
    return natsorted(files)


def pad_to_multiple(img_tensor, multiple=8):
    """
    将图像 pad 到 multiple 的倍数
    返回: (padded_tensor, original_h, original_w)
    """
    _, _, h, w = img_tensor.shape
    H = ((h + multiple) // multiple) * multiple
    W = ((w + multiple) // multiple) * multiple
    padh = H - h if h % multiple != 0 else 0
    padw = W - w if w % multiple != 0 else 0
    
    if padh > 0 or padw > 0:
        img_tensor = F.pad(img_tensor, (0, padw, 0, padh), mode='reflect')
    
    return img_tensor, h, w


def inference_full(model, input_tensor):
    """全图推理"""
    return model(input_tensor)


def inference_tile(model, input_tensor, tile_size, tile_overlap):
    """
    分块推理（用于处理大图）
    """
    b, c, h, w = input_tensor.shape
    tile = min(tile_size, h, w)
    
    assert tile % 8 == 0, "Tile size should be multiple of 8"
    
    stride = tile - tile_overlap
    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
    
    # 输出和权重累加器
    E = torch.zeros(b, c, h, w).type_as(input_tensor)
    W = torch.zeros_like(E)
    
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = input_tensor[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
            out_patch = model(in_patch)
            
            # 处理可能的 list 输出
            if isinstance(out_patch, list):
                out_patch = out_patch[-1]
            
            out_patch_mask = torch.ones_like(out_patch)
            E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
            W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
    
    # 平均重叠区域
    output = E.div_(W)
    
    return output


@torch.no_grad()
def process_image(model, img_path, output_dir, device, tile_size=None, tile_overlap=32):
    """处理单张图像"""
    img = load_img(img_path)
    
    # 转换为 tensor: (H, W, C) -> (1, C, H, W), 归一化到 [0, 1]
    input_tensor = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Pad 到 8 的倍数
    input_padded, padded_h, padded_w = pad_to_multiple(input_tensor, multiple=8)
    
    if tile_size is None:
        output = inference_full(model, input_padded)
    else:
        output = inference_tile(model, input_padded, tile_size, tile_overlap)
    
    # 处理可能的 list 输出
    if isinstance(output, list):
        output = output[-1]
    
    output = torch.clamp(output, 0, 1)
    output = output[:, :, :padded_h, :padded_w]
    
    # 转换回 numpy: (1, C, H, W) -> (H, W, C)
    output_np = output.permute(0, 2, 3, 1).cpu().detach().numpy()
    output_np = img_as_ubyte(output_np[0])
    
    filename = os.path.splitext(os.path.basename(img_path))[0]
    output_path = os.path.join(output_dir, f'{filename}.png')
    save_img(output_path, output_np)
    
    return output_path


def main():
    args = parse_args()
    
    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, using CPU')
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f'Using device: {device}')
    
    # 加载配置
    print(f'Loading config from: {args.config}')
    config = load_config(args.config)
    
    # 构建模型
    print('Building model...')
    model = build_model(config)
    model = load_weights(model, args.weights)
    model = model.to(device)
    model.eval()
    
    # 打印模型信息
    network_config = config['network_g']
    print(f"\nModel Config:")
    print(f"  - Type: {network_config.get('type', 'Restormer')}")
    print(f"  - Dim: {network_config.get('dim', 48)}")
    print(f"  - Num Blocks: {network_config.get('num_blocks', [4,6,6,8])}")
    print(f"  - Heads: {network_config.get('heads', [1,2,4,8])}")
    
    # 获取输入文件
    image_files = get_image_files(args.input_dir)
    if len(image_files) == 0:
        raise ValueError(f'No image files found in: {args.input_dir}')
    
    print(f'\nFound {len(image_files)} images to process')
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Output directory: {args.output_dir}')
    
    # 推理设置
    if args.tile:
        print(f'Using tile mode: tile_size={args.tile}, overlap={args.tile_overlap}')
    else:
        print('Using full resolution mode')
    
    print('\n' + '='*50)
    print('Starting inference...')
    print('='*50 + '\n')
    
    # 处理所有图像
    for img_path in tqdm(image_files, desc='Processing'):
        # 清理 GPU 缓存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        try:
            output_path = process_image(
                model=model,
                img_path=img_path,
                output_dir=args.output_dir,
                device=device,
                tile_size=args.tile,
                tile_overlap=args.tile_overlap
            )
        except Exception as e:
            print(f'\nError processing {img_path}: {e}')
            continue
    
    print(f'\n{"="*50}')
    print(f'Inference completed!')
    print(f'Results saved to: {args.output_dir}')
    print(f'{"="*50}')


if __name__ == '__main__':
    main()
     
