#!/usr/bin/env python3
import argparse
import os
import yaml
import torch
import models
import datasets
import utils
from models import DenoisingDiffusion, DiffusiveRestoration
from metrics_eval import evaluate_loader

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Latent-Retinex Diffusion Models Evaluation')
    parser.add_argument("--config", default='unsupervised.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--mode', type=str, default='evaluation', help='training or evaluation')
    parser.add_argument('--resume', default='ckpt/stage2/stage2_weight.pth.tar', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--image_folder", default='results/', type=str,
                        help="Location to save restored images")
    parser.add_argument("--paired", action="store_true", help="Set if the dataset is paired (supervised)")
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    return args, new_config

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def main():
    args, config = parse_args_and_config()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)
    config.device = device

    if torch.cuda.is_available():
        print("Note: Single GPU evaluation is supported.")

    print("=> Using dataset '{}'".format(config.data.val_dataset))
    DATASET = datasets.__dict__[config.data.type](config)
    _, val_loader = DATASET.get_loaders()

    print("=> Creating denoising-diffusion model...")
    diffusion = DenoisingDiffusion(args, config)
    restoration_model = DiffusiveRestoration(diffusion, args, config)
    
    # Evaluate using the new modular metrics evaluation function.
    results = evaluate_loader(val_loader, restoration_model, device, is_paired=args.paired)
    
    print("\nAverage Evaluation Metrics:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    main()
