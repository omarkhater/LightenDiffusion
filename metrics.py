import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips  # pip install lpips
import torch

# Create LPIPS model
lpips_model = lpips.LPIPS(net='alex')

def to_numpy(image_tensor):
    # Convert a torch.Tensor of shape [C, H, W] in [0, 1] to a numpy array [H, W, C] in [0, 255] as uint8.
    image_tensor = image_tensor.detach().cpu().clamp(0, 1)
    image_np = image_tensor.mul(255).byte().permute(1,2,0).numpy()
    return image_np

def compute_psnr(gt, pred):
    # gt and pred are numpy arrays of shape [H, W, C] in uint8.
    return peak_signal_noise_ratio(gt, pred, data_range=255)

def compute_ssim(gt, pred):
    # Use the new parameter channel_axis (set to -1) for multichannel images.
    return structural_similarity(gt, pred, data_range=255, channel_axis=-1, win_size=7)

def compute_lpips(gt, pred):
    # Ensure inputs have a batch dimension.
    if gt.dim() == 3:
        gt = gt.unsqueeze(0)
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    # Normalize inputs from [0, 1] to [-1, 1]
    gt_norm = gt * 2 - 1
    pred_norm = pred * 2 - 1
    # Move inputs to the same device as lpips_model.
    device = next(lpips_model.parameters()).device
    gt_norm = gt_norm.to(device)
    pred_norm = pred_norm.to(device)
    with torch.no_grad():
        dist = lpips_model(gt_norm, pred_norm)
    return dist.item()

import pyiqa

def compute_niqe(image_tensor):
    # image_tensor should be a torch.Tensor of shape [1, C, H, W] in [0, 1].
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    niqe_metric = pyiqa.create_metric('niqe_matlab')
    return niqe_metric(image_tensor).item()



def compute_pi(lpips_val, niqe_val):
    # A common definition of PI is the average of LPIPS and NIQE.
    return 0.5 * (lpips_val + niqe_val)
