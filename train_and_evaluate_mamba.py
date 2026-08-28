"""
Full End-to-End MambaPFAN Training and Visual Evaluation Script
===============================================================
Trains Mamba-PFAN on GPU, saves checkpoints, and generates real de-smoked visual outputs.
Compatible with Local GPUs (RTX 5060/4060), Kaggle (T4/P100), and Google Colab.
"""

import os
import sys
import time
import copy
import math
import torch
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# Unbuffered line printing
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# If no CLI arguments provided, apply intelligent defaults based on environment
if len(sys.argv) <= 1:
    is_kaggle = os.path.exists('/kaggle')
    is_colab = os.path.exists('/content')

    if is_kaggle:
        dataroot = '/kaggle/input/desmoking-dataset/composite' if os.path.exists('/kaggle/input/desmoking-dataset/composite') else './datasets/composite'
        checkpoints_dir = '/kaggle/working/checkpoints'
        num_threads = '4'
        batch_size = '8'
    elif is_colab:
        dataroot = '/content/datasets/composite' if os.path.exists('/content/datasets/composite') else './datasets/composite'
        checkpoints_dir = '/content/checkpoints'
        num_threads = '2'
        batch_size = '8'
    else:
        dataroot = './datasets/composite'
        checkpoints_dir = './checkpoints'
        num_threads = '0'
        batch_size = '4'

    sys.argv = [
        'train.py',
        '--dataroot', dataroot,
        '--name', 'mamba_Final',
        '--model', 'pix2pix',
        '--netG', 'mamba_pfan',
        '--netD', 'basic',
        '--direction', 'AtoB',
        '--dataset_mode', 'aligned',
        '--norm', 'batch',
        '--checkpoints_dir', checkpoints_dir,
        '--embed_dim', '64',
        '--ndf', '64',
        '--ngf', '64',
        '--gpu_ids', '0',
        '--batch_size', batch_size,
        '--display_id', '-1',
        '--num_threads', num_threads,
        '--n_epochs', '30',
        '--n_epochs_decay', '0',
        '--print_freq', '20'
    ]

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util import util


def main():
    print("=" * 65, flush=True)
    print("MAMBA-PFAN: TRAINING & DE-SMOKING PIPELINE INITIALIZING", flush=True)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_name}", flush=True)
    print("=" * 65, flush=True)

    # 1. Parse Training Options
    opt = TrainOptions().parse()

    # 2. Create Train Dataset & Model
    train_dataset = create_dataset(opt)
    dataset_size = len(train_dataset)
    print(f"Training images: {dataset_size}", flush=True)

    model = create_model(opt)
    model.setup(opt)

    total_iters = 0
    num_epochs = opt.n_epochs + opt.n_epochs_decay

    # 3. Training Loop
    for epoch in range(opt.epoch_count, num_epochs + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        print(f"\n--- Starting Epoch {epoch}/{num_epochs} ---", flush=True)

        for i, data in enumerate(train_dataset):
            iter_start = time.time()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if (i + 1) % opt.print_freq == 0 or (i + 1) == len(train_dataset):
                losses = model.get_current_losses()
                loss_str = " | ".join([f"{k}: {v:.3f}" for k, v in losses.items()])
                step_time = (time.time() - iter_start) / opt.batch_size
                print(f"[Epoch {epoch}/{num_epochs}] Step {i+1}/{len(train_dataset)} ({step_time:.3f}s/item) -> {loss_str}", flush=True)

        model.update_learning_rate()
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_time:.1f}s. Saving checkpoints...", flush=True)
        model.save_networks(epoch)
        model.save_networks('latest')

    print("\n" + "=" * 65, flush=True)
    print("TRAINING FINISHED! RUNNING TEST INFERENCE & VISUAL GENERATION...", flush=True)
    print("=" * 65, flush=True)

    # 4. Run Test Inference and Generate Visual Comparisons
    val_opt = copy.deepcopy(opt)
    val_opt.phase = 'test'
    val_opt.isTrain = False
    val_opt.serial_batches = True
    val_opt.no_flip = True
    val_opt.max_dataset_size = 20
    test_dataset = create_dataset(val_opt)

    output_dir = os.path.join(opt.checkpoints_dir, opt.name, 'results', 'mamba_Final', 'visual_comparisons')
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    psnr_list, ssim_list, mse_list = [], [], []

    with torch.no_grad():
        for i, data in enumerate(test_dataset):
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()[0]
            short_name = os.path.splitext(os.path.basename(img_path))[0]

            real_A = util.tensor2im(visuals['real_A'])
            fake_B = util.tensor2im(visuals['fake_B'])
            real_B = util.tensor2im(visuals['real_B'])

            # Compute accurate PSNR & SSIM metrics
            psnr_val = compare_psnr(real_B, fake_B, data_range=255)
            ssim_val = compare_ssim(real_B, fake_B, channel_axis=2, data_range=255)
            mse_val = np.mean((real_B.astype(np.float64) - fake_B.astype(np.float64)) ** 2)

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            mse_list.append(mse_val)

            # Create 3-panel comparison: [ Smoky Input | MambaPFAN De-Smoked | Ground Truth Clean ]
            h, w, c = real_A.shape
            side_by_side = np.zeros((h, w * 3, c), dtype=np.uint8)
            side_by_side[:, :w, :] = real_A
            side_by_side[:, w:2*w, :] = fake_B
            side_by_side[:, 2*w:, :] = real_B

            comp_path = os.path.join(output_dir, f"{short_name}_comparison.png")
            Image.fromarray(side_by_side).save(comp_path)

            desmoked_path = os.path.join(output_dir, f"{short_name}_desmoked.png")
            Image.fromarray(fake_B).save(desmoked_path)

            print(f"Sample {short_name}: PSNR = {psnr_val:.2f} dB, SSIM = {ssim_val:.4f} -> Saved {short_name}_comparison.png", flush=True)

    avg_psnr = np.mean(psnr_list) if psnr_list else 0.0
    avg_ssim = np.mean(ssim_list) if ssim_list else 0.0
    avg_mse = np.mean(mse_list) if mse_list else 0.0

    summary_file = os.path.join(output_dir, 'evaluation_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("MambaPFAN Test Results\n")
        f.write("======================\n")
        f.write(f"Number of test images: {len(psnr_list)}\n")
        f.write(f"Average PSNR: {avg_psnr:.4f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"Average MSE:  {avg_mse:.4f}\n")

    print("\n" + "=" * 65, flush=True)
    print("MAMBA-PFAN EVALUATION COMPLETE!", flush=True)
    print(f"Average PSNR: {avg_psnr:.4f} dB", flush=True)
    print(f"Average SSIM: {avg_ssim:.4f}", flush=True)
    print(f"Visual outputs saved to: {output_dir}", flush=True)
    print("=" * 65, flush=True)


if __name__ == '__main__':
    main()
