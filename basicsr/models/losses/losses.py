import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from torch.autograd import Variable
import torchvision.models as models

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)


class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)


class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, x, y, weight=None):
        diff = x - y
        loss = torch.sqrt((diff * diff) + (self.eps*self.eps))
        if weight is not None:
            loss = loss * weight
        
        if self.reduction == 'mean':
            return self.loss_weight * torch.mean(loss)
        elif self.reduction == 'sum':
            return self.loss_weight * torch.sum(loss)
        else:
            return self.loss_weight * loss

class SSIMLoss(nn.Module):
    """SSIM Loss.
    
    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        window_size (int): Window size for SSIM. Default: 11.
    """

    def __init__(self, loss_weight=1.0, window_size=11, reduction='mean'):
        super(SSIMLoss, self).__init__()
        self.loss_weight = loss_weight
        self.window_size = window_size
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def forward(self, img1, img2, weight=None):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        
        # SSIM Loss doesn't support weight map easily in this implementation
        # But for interface consistency we accept it
        return self.loss_weight * (1 - self._ssim(img1, img2, window, self.window_size, channel))

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

class GradientLoss(nn.Module):
    """Gradient Loss to preserve edges and textures."""
    def __init__(self, loss_weight=1.0):
        super(GradientLoss, self).__init__()
        self.loss_weight = loss_weight
        # Sobel kernel
        kernel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, pred, target, weight=None):
        # Apply to each channel
        b, c, h, w = pred.shape
        # Flatten channels to use conv2d with groups=c
        # Or just apply to each channel separately. Simple way: reshape (B*C, 1, H, W)
        pred_flat = pred.reshape(b*c, 1, h, w)
        target_flat = target.reshape(b*c, 1, h, w)
        
        # Conv
        pred_gx = F.conv2d(pred_flat, self.kernel_x, padding=1)
        pred_gy = F.conv2d(pred_flat, self.kernel_y, padding=1)
        
        target_gx = F.conv2d(target_flat, self.kernel_x, padding=1)
        target_gy = F.conv2d(target_flat, self.kernel_y, padding=1)
        
        loss_x = torch.abs(pred_gx - target_gx)
        loss_y = torch.abs(pred_gy - target_gy)
        loss = loss_x + loss_y
        
        if weight is not None:
            # weight shape (B, 1, H, W) -> (B*C, 1, H, W)
            # Need to match dimensions carefully
            # Assume weight is same for all channels
            weight_flat = weight.repeat(1, c, 1, 1).reshape(b*c, 1, h, w)
            loss = loss * weight_flat
            
        return self.loss_weight * torch.mean(loss)

class ColorLoss(nn.Module):
    """Color consistency loss using simple Gaussian blur to compare low-freq color info."""
    def __init__(self, loss_weight=1.0, patch_size=16):
        super(ColorLoss, self).__init__()
        self.loss_weight = loss_weight
        self.patch_size = patch_size
        self.pool = nn.AvgPool2d(patch_size)

    def forward(self, pred, target):
        # Average color in patches
        pred_mean = self.pool(pred)
        target_mean = self.pool(target)
        loss = F.mse_loss(pred_mean, target_mean)
        return self.loss_weight * loss

class PerceptualLoss(nn.Module):
    """VGG-based Perceptual Loss."""
    def __init__(self, layer_weights=None, vgg_type='vgg19', use_input_norm=True, range_norm=False, loss_weight=1.0):
        super(PerceptualLoss, self).__init__()
        self.loss_weight = loss_weight
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        if layer_weights is None:
            layer_weights = {'conv5_4': 1.0}
        self.layer_weights = layer_weights

        # VGG model
        if vgg_type == 'vgg19':
            vgg = models.vgg19(pretrained=True)
        elif vgg_type == 'vgg16':
            vgg = models.vgg16(pretrained=True)
        else:
            raise ValueError(f'Unsupported VGG type: {vgg_type}')

        # Extract features
        self.vgg_layers = vgg.features
        self.layer_name_mapping = {
            '3': "conv1_2",
            '8': "conv2_2",
            '17': "conv3_4",
            '26': "conv4_4",
            '35': "conv5_4"
        }
        
        # We only need layers up to the max index we use
        # Find max index
        max_idx = 0
        for name in layer_weights.keys():
            # Find index corresponding to name
            for k, v in self.layer_name_mapping.items():
                if v == name:
                    max_idx = max(max_idx, int(k))
        
        self.vgg_layers = nn.Sequential(*list(vgg.features.children())[:max_idx+1])
        
        # Freeze VGG parameters
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
            
        # VGG Mean/Std for normalization (ImageNet)
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, gt):
        # Input x, gt should be in [0, 1]
        
        if self.range_norm:
            x = (x + 1) / 2
            gt = (gt + 1) / 2

        if self.use_input_norm:
            x = (x - self.mean) / self.std
            gt = (gt - self.mean) / self.std

        loss = 0.0
        x_features = x
        gt_features = gt
        
        for name, module in self.vgg_layers.named_children():
            x_features = module(x_features)
            gt_features = module(gt_features)
            
            if name in self.layer_name_mapping:
                layer_name = self.layer_name_mapping[name]
                if layer_name in self.layer_weights:
                    loss += F.l1_loss(x_features, gt_features) * self.layer_weights[layer_name]
        
        return self.loss_weight * loss

class GTMeanLoss(nn.Module):
    """GT-Mean Loss for tackling brightness mismatch."""
    def __init__(self, loss_weight=1.0, sigma=0.1, eps=1e-3):
        super(GTMeanLoss, self).__init__()
        self.loss_weight = loss_weight
        self.sigma = sigma
        self.eps = 1e-6
        # Use Charbonnier Loss (smooth L1) as the base loss for better convergence
        self.charbonnier_eps = eps

    def get_mean(self, x):
        # Calculate mean brightness of the whole image
        return x.mean(dim=[1, 2, 3], keepdim=True)

    def get_bhattacharyya_distance(self, mu1, mu2):
        # Assuming sigma = sigma * mean
        sigma1 = self.sigma * mu1
        sigma2 = self.sigma * mu2
        
        var1 = sigma1 ** 2
        var2 = sigma2 ** 2
        var_avg = (var1 + var2) / 2
        
        term1 = (mu1 - mu2) ** 2 / (4 * var_avg + self.eps)
        term2 = 0.5 * torch.log((var_avg + self.eps) / (torch.sqrt(var1 * var2) + self.eps))
        
        return term1 + term2

    def base_loss_fn(self, x, y, weight=None):
        # Implementation of Charbonnier Loss logic
        diff = x - y
        loss = torch.sqrt((diff * diff) + (self.charbonnier_eps * self.charbonnier_eps))
        if weight is not None:
            loss = loss * weight
        return loss

    def forward(self, pred, target, weight=None):
        # 1. Calculate Means
        mu_pred = self.get_mean(pred)
        mu_target = self.get_mean(target)
        
        # 2. Calculate Weight W
        db = self.get_bhattacharyya_distance(mu_target, mu_pred)
        W = torch.clamp(db, 0, 1)
        W = W.detach() # Detach W as per paper
        
        # 3. Original Loss
        l_orig = self.base_loss_fn(pred, target, weight)
        l_orig = l_orig.mean()
        
        # 4. Aligned Loss
        # lambda = E[y] / E[f(x)]
        lambda_scale = mu_target / (mu_pred + self.eps)
        pred_aligned = pred * lambda_scale
        
        l_align = self.base_loss_fn(pred_aligned, target, weight)
        l_align = l_align.mean()

        # 5. Combined Loss
        loss = W * l_orig + (1 - W) * l_align
        
        return self.loss_weight * loss
