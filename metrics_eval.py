#!/usr/bin/env python3
import os
import torch
import numpy as np
import torch.nn.functional as F
from metrics import to_numpy, compute_psnr, compute_ssim, compute_lpips, compute_niqe, compute_pi
import utils

def evaluate_loader(val_loader, restoration_model, device, is_paired=True):
    """
    Evaluate the restoration model on a validation DataLoader.
    Uses the forward_sample() method of DiffusiveRestoration for processing.

    Args:
        val_loader: DataLoader yielding (x, y) with x of shape [B, 6, H, W].
        restoration_model: Instance of DiffusiveRestoration.
        device: torch.device for computation.
        is_paired: Boolean flag; if True, ground-truth is available.
        
    Returns:
        results (dict): Dictionary of average metrics.
    """
    psnr_total, ssim_total, lpips_total = 0.0, 0.0, 0.0
    niqe_total, pi_total = 0.0, 0.0
    count = 0

    restoration_model.diffusion.model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            print(f"evaluating {i+1}/{len(val_loader)}...")
            # x is expected to be [B, 6, H, W]. For paired data, gt is the last 3 channels.
            low_img = x[:, :3, :, :].to(device)
            if is_paired:
                gt_img = x[:, 3:, :, :].to(device)
            else:
                gt_img = None

            # Debug prints
            #print(f"x shape: {x.shape}, y: {y}")
            #print(f"low_img shape: {low_img.shape}")

            b, c, h, w = low_img.shape
            if c != 3:
                print(f"Warning: Expected 3 channels in low image, got {c}. Skipping sample {y[0]}.")
                continue

            # Use the shared forward_sample method to get the prediction.
            pred_img = restoration_model.forward_sample(x)
            
            # Save the restored image.
            save_dir = os.path.join(restoration_model.args.image_folder, restoration_model.config.data.val_dataset)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{y[0]}")
            utils.logging.save_image(pred_img, save_path)
            print(f"Processed image {y[0]}")

            # Compute metrics.
            if is_paired:
                gt_np = to_numpy(gt_img[0])
                pred_np = to_numpy(pred_img[0])
                psnr = compute_psnr(gt_np, pred_np)
                ssim = compute_ssim(gt_np, pred_np)
                lpips_val = compute_lpips(gt_img[0], pred_img[0])
                psnr_total += psnr
                ssim_total += ssim
                lpips_total += lpips_val
                print(f"[{y[0]}] Supervised: PSNR={psnr:.2f}, SSIM={ssim:.4f}, LPIPS={lpips_val:.4f}", end=", ")
            else:
                lpips_val = compute_lpips(low_img[0], pred_img[0])
                print(f"[{y[0]}] Unpaired: LPIPS={lpips_val:.4f}", end=", ")
            
            niqe_val = compute_niqe(pred_img[0])
            pi_val = compute_pi(lpips_val, niqe_val)
            print(f"NIQE={niqe_val:.4f}, PI={pi_val:.4f}")
            
            niqe_total += niqe_val
            pi_total += pi_val
            count += 1

    results = {}
    if count > 0:
        if is_paired:
            results["PSNR"] = psnr_total / count
            results["SSIM"] = ssim_total / count
            results["LPIPS"] = lpips_total / count
        results["NIQE"] = niqe_total / count
        results["PI"] = pi_total / count
    return results
