#!/usr/bin/env python3
import torch
import models
import datasets
import utils
from models import DenoisingDiffusion, DiffusiveRestoration
from metrics_eval import evaluate_loader
from utils.config_utils import parse_args_and_config

def main():
    args, config = parse_args_and_config(mode="evaluation")
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

    results = evaluate_loader(val_loader, restoration_model, device, is_paired=args.paired)
    
    print("\nAverage Evaluation Metrics:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    main()
