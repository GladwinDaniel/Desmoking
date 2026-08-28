"""
Evaluation and Visual Comparison Script for MambaPFAN
=====================================================
Loads trained MambaPFAN checkpoint, runs inference on test images,
computes PSNR / SSIM / MSE metrics, and saves side-by-side visual comparisons.
"""

import os
import sys
import time
import math
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.mamba_pfan import MambaPFAN
from models.networks import get_norm_layer


def evaluate(
    checkpoint_path='./checkpoints/mamba_Final/latest_net_G.pth',
    test_dir='./datasets/composite/test',
    output_dir='./results/mamba_eval',
    image_size=(256, 256),
    device=None
):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Running evaluation on device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Initialize MambaPFAN Model
    norm_layer = get_norm_layer(norm_type='batch')
    netG = MambaPFAN(
        input_nc=3,
        output_nc=3,
        ngf=64,
        hidden_dim=64,
        layers=[2, 2, 2],
        d_state=16,
        d_conv=4,
        expand=2,
        norm_layer_1=norm_layer
    ).to(device)

    # 2. Load Weights
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    weights = torch.load(checkpoint_path, map_location=device)
    netG.load_state_dict(weights)
    netG.eval()
    print(f"Loaded generator checkpoint: {checkpoint_path}")

    # 3. Define Image Transform
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 4. Locate Test Images (Supports paired subfolders 'A' & 'B', or single folder)
    testA_dir = os.path.join(test_dir, 'A') if os.path.isdir(os.path.join(test_dir, 'A')) else test_dir
    testB_dir = os.path.join(test_dir, 'B') if os.path.isdir(os.path.join(test_dir, 'B')) else None

    valid_extensions = ('.bmp', '.png', '.jpg', '.jpeg')
    image_files = sorted([f for f in os.listdir(testA_dir) if f.lower().endswith(valid_extensions)])
    print(f"Found {len(image_files)} test images in {testA_dir}")

    psnr_list, ssim_list, mse_list = [], [], []

    with torch.no_grad():
        for i, fname in enumerate(image_files):
            pathA = os.path.join(testA_dir, fname)
            imgA = Image.open(pathA).convert('RGB')
            tA = transform(imgA).unsqueeze(0).to(device)

            fakeB = netG(tA)

            # Convert network output tensor [-1, 1] to uint8 [0, 255]
            out = fakeB.squeeze(0).cpu().float().numpy()
            out = (np.transpose(out, (1, 2, 0)) + 1) / 2.0 * 255.0
            out = np.clip(out, 0, 255).astype(np.uint8)

            # Save individual de-smoked output
            short_name = os.path.splitext(fname)[0]
            desmoked_path = os.path.join(output_dir, f"{short_name}_desmoked.png")
            Image.fromarray(out).save(desmoked_path)

            # If ground truth B exists, compute metrics and side-by-side comparison
            has_gt = testB_dir and os.path.exists(os.path.join(testB_dir, fname))
            if has_gt:
                imgB = Image.open(os.path.join(testB_dir, fname)).convert('RGB')
                gt = imgB.resize(image_size, Image.BICUBIC)
                gt_arr = np.array(gt)

                in_arr = np.array(imgA.resize(image_size, Image.BICUBIC))

                mse = np.mean((gt_arr.astype(np.float64) - out.astype(np.float64)) ** 2)
                psnr = compare_psnr(gt_arr, out, data_range=255)
                ssim = compare_ssim(gt_arr, out, channel_axis=2, data_range=255)

                psnr_list.append(psnr)
                ssim_list.append(ssim)
                mse_list.append(mse)

                # 3-panel: [ Smoky Input | Mamba De-Smoked | Clean Ground Truth ]
                h, w, c = in_arr.shape
                comp = np.zeros((h, w * 3, c), dtype=np.uint8)
                comp[:, :w, :] = in_arr
                comp[:, w:2*w, :] = out
                comp[:, 2*w:, :] = gt_arr

                comp_path = os.path.join(output_dir, f"{short_name}_comparison.png")
                Image.fromarray(comp).save(comp_path)

            if (i + 1) % 20 == 0 or (i + 1) == len(image_files):
                if psnr_list:
                    print(f"[{i+1}/{len(image_files)}] - PSNR: {np.mean(psnr_list):.2f} dB | SSIM: {np.mean(ssim_list):.4f}")
                else:
                    print(f"[{i+1}/{len(image_files)}] processed.")

    if psnr_list:
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        avg_mse = np.mean(mse_list)

        summary_file = os.path.join(output_dir, 'evaluation_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("MambaPFAN Evaluation Summary\n")
            f.write("============================\n")
            f.write(f"Number of test images: {len(psnr_list)}\n")
            f.write(f"Average PSNR: {avg_psnr:.4f} dB\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
            f.write(f"Average MSE:  {avg_mse:.4f}\n")

        print("\n" + "=" * 55)
        print("EVALUATION COMPLETE")
        print(f"Average PSNR: {avg_psnr:.4f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average MSE:  {avg_mse:.4f}")
        print(f"Outputs saved to: {output_dir}")
        print("=" * 55)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Mamba-PFAN on test images")
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/mamba_Final/latest_net_G.pth', help='Path to generator .pth file')
    parser.add_argument('--test_dir', type=str, default='./datasets/composite/test', help='Path to test folder (contains A and B or images)')
    parser.add_argument('--output_dir', type=str, default='./results/mamba_eval', help='Output directory for generated images')
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        test_dir=args.test_dir,
        output_dir=args.output_dir
    )
