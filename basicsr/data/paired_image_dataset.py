from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP

import random
import numpy as np
import torch
import cv2
import os
class Dataset_PairedImage(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']
            
        # Initialize mask paths
        self.mask_paths = {}
        mask_names = ['hairmask', 'personmask', 'skinmask', 'skinmaskFace', 'skymask']
        import os
        if 'dataroot_lq' in opt:
            input_dir = opt['dataroot_lq']
            # Assume masks are in flash_data/mask_{name}
            # input_dir is likely flash_data/base_1k or flash_data/input
            # parent is flash_data
            root_dir = os.path.dirname(input_dir.rstrip('/'))
            
            for name in mask_names:
                mask_dir = os.path.join(root_dir, f'mask_{name}')
                if os.path.exists(mask_dir):
                    self.mask_paths[name] = mask_dir
                    print(f"[Dataset] Found mask folder: {mask_dir}")
                else:
                    print(f"[Dataset] Warning: Mask folder not found: {mask_dir}. Root dir was: {root_dir}")

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))
        if img_gt is None:
            raise Exception("gt path {} read None".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))
        if img_lq is None:
            raise Exception("lq path {} read None".format(lq_path))
        
        # 确保尺寸是 8 的倍数（Restormer 有多层下采样，需要尺寸能被 8 整除）
        h, w = img_lq.shape[:2]
        new_h = (h // 8) * 8
        new_w = (w // 8) * 8
        if new_h != h or new_w != w:
            img_lq = cv2.resize(img_lq, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 确保 gt 和 lq 尺寸一致
        target_h, target_w = img_lq.shape[:2]
        gt_h, gt_w = img_gt.shape[:2]
        if gt_h != target_h or gt_w != target_w:
            img_gt = cv2.resize(img_gt, (target_w, target_h), interpolation=cv2.INTER_AREA)
            
        # Load Masks
        masks = []
        mask_names = ['hairmask', 'personmask', 'skinmask', 'skinmaskFace', 'skymask']
        filename = os.path.basename(lq_path)
        base_name, _ = os.path.splitext(filename)
        
        for name in mask_names:
            if name in self.mask_paths:
                # Try original extension first
                mask_path = os.path.join(self.mask_paths[name], filename)
                if not os.path.exists(mask_path):
                    # Try png
                    mask_path = os.path.join(self.mask_paths[name], base_name + '.png')
                if not os.path.exists(mask_path):
                    # Try bmp
                    mask_path = os.path.join(self.mask_paths[name], base_name + '.bmp')
                
                if os.path.exists(mask_path):
                    # Read as grayscale
                    m_bytes = self.file_client.get(mask_path, 'lq') # use 'lq' bucket or generic
                    m = imfrombytes(m_bytes, flag='grayscale', float32=True)
                    if m is None:
                        m = np.zeros((target_h, target_w), dtype=np.float32)
                    else:
                        if m.ndim == 3:
                            m = m[:,:,0]
                        # Resize mask to match image size if different
                        if m.shape[0] != target_h or m.shape[1] != target_w:
                            m = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                else:
                    m = np.zeros((target_h, target_w), dtype=np.float32)
            else:
                m = np.zeros((target_h, target_w), dtype=np.float32)
            
            # Add channel dim: (H, W, 1)
            masks.append(np.expand_dims(m, axis=2))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            
            # Stack images for synchronized processing
            # img_gt: HWC (3), img_lq: HWC (3), masks: list of HWC (1)
            # Concatenate all: (H, W, 3+3+5)
            mask_stack = np.concatenate(masks, axis=2)
            all_stack = np.concatenate([img_gt, img_lq, mask_stack], axis=2)
            
            if gt_size > 0:
                # padding: 确保图像尺寸不小于 gt_size
                h, w, _ = all_stack.shape
                h_pad = max(0, gt_size - h)
                w_pad = max(0, gt_size - w)
                if h_pad > 0 or w_pad > 0:
                    all_stack = cv2.copyMakeBorder(all_stack, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)

                # random crop
                # paired_random_crop(img_gts, img_lqs, ...)
                # It returns (patch_gts, patch_lqs).
                # We can cheat by passing all_stack as both? No.
                # We need to implement manual random crop to ensure sync.
                
                h, w, _ = all_stack.shape
                if h > gt_size or w > gt_size:
                    # Random Crop
                    # From basicsr code logic:
                    x = random.randint(0, h - gt_size)
                    y = random.randint(0, w - gt_size)
                    all_stack = all_stack[x:x+gt_size, y:y+gt_size, :]

            # flip, rotation
            if self.geometric_augs:
                # random_augmentation expects list
                # returns list
                out_list = random_augmentation(all_stack)
                all_stack = out_list[0]
                
            # Unpack
            img_gt = all_stack[:, :, 0:3]
            img_lq = all_stack[:, :, 3:6]
            mask_stack = all_stack[:, :, 6:]
            
            # Update masks list
            masks = [mask_stack[:, :, i:i+1] for i in range(mask_stack.shape[2])]

        # 强制转换为 3 通道 BGR，避免 cvtColor 报错
        if len(img_gt.shape) == 2:
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_GRAY2BGR)
        elif img_gt.shape[2] == 4:
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGRA2BGR)
            
        if len(img_lq.shape) == 2:
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_GRAY2BGR)
        elif img_lq.shape[2] == 4:
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGRA2BGR)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
                                    
        # Masks to tensor (HWC -> CHW, no RGB conversion)
        # img2tensor handles list of numpy arrays
        mask_tensors = img2tensor(masks, bgr2rgb=False, float32=True)
        
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        
        return_dict = {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }
        
        # Add masks to dict
        for i, name in enumerate(mask_names):
            return_dict[f'mask_{name}'] = mask_tensors[i]
            
        return return_dict

    def __len__(self):
        return len(self.paths)

class Dataset_GaussianDenoising(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_GaussianDenoising, self).__init__()
        self.opt = opt

        if self.opt['phase'] == 'train':
            self.sigma_type  = opt['sigma_type']
            self.sigma_range = opt['sigma_range']
            assert self.sigma_type in ['constant', 'random', 'choice']
        else:
            self.sigma_test = opt['sigma_test']
        self.in_ch = opt['in_ch']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None        

        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [
                    osp.join(self.gt_folder,
                             line.split(' ')[0]) for line in fin
                ]
        else:
            from basicsr.utils.scandir import scandir
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')

        if self.in_ch == 3:
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        else:
            try:
                img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = img_gt.copy()


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            
            # 只有当 gt_size > 0 时才进行 cropping/padding
            if gt_size > 0:
                # padding
                img_gt, img_lq = padding(img_gt, img_lq, gt_size)

                # random crop
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            # BGR to RGB, HWC to CHW, numpy to tensor
            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=False,
                                        float32=True)

            if self.sigma_type == 'constant':
                sigma_value = self.sigma_range
            elif self.sigma_type == 'random':
                sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
            elif self.sigma_type == 'choice':
                sigma_value = random.choice(self.sigma_range)

            noise_level = torch.FloatTensor([sigma_value])/255.0
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)

        else:            
            np.random.seed(seed=0)
            img_lq += np.random.normal(0, self.sigma_test/255.0, img_lq.shape)

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                            bgr2rgb=False,
                            float32=True)
            img_lq = torch.tensor(img_lq, dtype=torch.float32) # Ensure tensor
            
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

class Dataset_DefocusDeblur_DualPixel_16bit(data.Dataset):
    def __init__(self, opt):
        super(Dataset_DefocusDeblur_DualPixel_16bit, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_folder, self.lqR_folder = opt['dataroot_gt'], opt['dataroot_lqL'], opt['dataroot_lqR']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = paired_DP_paths_from_folder(
            [self.lqL_folder, self.lqR_folder, self.gt_folder], ['lqL', 'lqR', 'gt'],
            self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lqL_path = self.paths[index]['lqL_path']
        img_bytes = self.file_client.get(lqL_path, 'lqL')
        try:
            img_lqL = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqL path {} not working".format(lqL_path))

        lqR_path = self.paths[index]['lqR_path']
        img_bytes = self.file_client.get(lqR_path, 'lqR')
        try:
            img_lqR = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqR path {} not working".format(lqR_path))


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_lqL, img_lqR, img_gt = padding_DP(img_lqL, img_lqR, img_gt, gt_size)

            # random crop
            img_lqL, img_lqR, img_gt = paired_random_crop_DP(img_lqL, img_lqR, img_gt, gt_size, scale, gt_path)
            
            # flip, rotation            
            if self.geometric_augs:
                img_lqL, img_lqR, img_gt = random_augmentation(img_lqL, img_lqR, img_gt)
        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lqL, img_lqR, img_gt = img2tensor([img_lqL, img_lqR, img_gt],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lqL, self.mean, self.std, inplace=True)
            normalize(img_lqR, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        img_lq = torch.cat([img_lqL, img_lqR], 0)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lqL_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)