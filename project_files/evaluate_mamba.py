"""
Evaluation and Visual Comparison Script for MambaPFAN
=====================================================
Loads trained MambaPFAN checkpoint, runs inference on test images,
computes PSNR / SSIM metrics, and saves side-by-side visual comparisons.
"""

import os
import time
import math
import torch
import numpy as np
from PIL import Image
from collections import OrderedDict
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util


def evaluate(opt):
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    output_dir = os.path.join(opt.results_dir, opt.name, 'visual_comparisons')
    os.makedirs(output_dir, exist_ok=True)

    psnr_list, ssim_list, mse_list = [], [], []

    print(f"Running evaluation on {len(dataset)} test samples...")

    with torch.no_grad():
        for i, data in enumerate(dataset):
            if i >= opt.num_test:
                break
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()[0]
            short_name = os.path.splitext(os.path.basename(img_path))[0]

            # Convert to numpy uint8
            if 'fake_B' in visuals and 'real_B' in visuals:
                real_A = util.tensor2im(visuals['real_A'])
                fake_B = util.tensor2im(visuals['fake_B'])
                real_B = util.tensor2im(visuals['real_B'])
            else:
                images = [util.tensor2im(v) for v in visuals.values()]
                real_A = images[0]
                fake_B = images[1]
                real_B = images[2] if len(images) > 2 else images[0]

            # Metrics
            psnr_val = compare_psnr(real_B, fake_B, data_range=255)
            ssim_val = compare_ssim(real_B, fake_B, channel_axis=2, data_range=255)
            mse_val = np.mean((real_B.astype(np.float64) - fake_B.astype(np.float64)) ** 2)

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            mse_list.append(mse_val)

            # Create side-by-side comparison: [Smoky Input | MambaPFAN Output | Ground Truth Clean]
            h, w, c = real_A.shape
            side_by_side = np.zeros((h, w * 3, c), dtype=np.uint8)
            side_by_side[:, :w, :] = real_A
            side_by_side[:, w:2*w, :] = fake_B
            side_by_side[:, 2*w:, :] = real_B

            comp_img = Image.fromarray(side_by_side)
            comp_path = os.path.join(output_dir, f"{short_name}_comparison.png")
            comp_img.save(comp_path)

            # Also save individual de-smoked output
            clean_output_path = os.path.join(output_dir, f"{short_name}_desmoked.png")
            Image.fromarray(fake_B).save(clean_output_path)

            if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
                print(f"[{i+1}/{len(dataset)}] - PSNR: {np.mean(psnr_list):.3f} dB | SSIM: {np.mean(ssim_list):.4f}")

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_mse = np.mean(mse_list)

    summary_file = os.path.join(output_dir, 'evaluation_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"MambaPFAN Test Results\n")
        f.write(f"======================\n")
        f.write(f"Number of test images: {len(psnr_list)}\n")
        f.write(f"Average PSNR: {avg_psnr:.4f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"Average MSE:  {avg_mse:.4f}\n")

    print("\n" + "=" * 40)
    print(f"MAMBA-PFAN EVALUATION COMPLETE")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average MSE:  {avg_mse:.4f}")
    print(f"Visual outputs saved to: {output_dir}")
    print("=" * 40)


if __name__ == '__main__':
    opt = TestOptions().parse()
    evaluate(opt)
