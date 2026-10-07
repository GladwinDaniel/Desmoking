"""Evaluate PFAN-family generators with one reproducible metric protocol."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torchvision.transforms as transforms

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.networks import define_G


VALID_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg"}


def resolve_pairs(test_dir):
    """Resolve paired input/target directories without relying on frame order."""
    root = Path(test_dir)
    candidates = [
        (root / "A", root / "B"),
        (root / "testA", root / "testB"),
        (root, root / "B"),
    ]
    for input_dir, target_dir in candidates:
        if input_dir.is_dir() and target_dir.is_dir():
            inputs = {
                path.name: path
                for path in input_dir.iterdir()
                if path.suffix.lower() in VALID_EXTENSIONS
            }
            targets = {
                path.name: path
                for path in target_dir.iterdir()
                if path.suffix.lower() in VALID_EXTENSIONS
            }
            names = sorted(inputs.keys() & targets.keys())
            if names:
                return [(name, inputs[name], targets[name]) for name in names]
    raise FileNotFoundError(
        f"No paired images found below {root}. Expected A/B or testA/testB directories."
    )


def load_generator(model_name, checkpoint_path, device, ngf=64, scan_mode="four_way"):
    net = define_G(
        opt=None,
        input_nc=3,
        output_nc=3,
        ngf=ngf,
        netG=model_name,
        norm="batch",
        init_type="normal",
        gpu_ids=[],
        scan_mode=scan_mode,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    state_dict = {
        key.removeprefix("module."): value for key, value in checkpoint.items()
    }
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    return net


def image_to_tensor(image, image_size, device):
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform(image).unsqueeze(0).to(device)


def tensor_to_image(tensor):
    array = tensor.squeeze(0).detach().cpu().numpy()
    array = np.transpose(array, (1, 2, 0))
    return np.clip((array + 1.0) * 127.5, 0, 255).astype(np.uint8)


def compute_metrics(output, target):
    output_float = output.astype(np.float64)
    target_float = target.astype(np.float64)
    lab_output = rgb2lab(output.astype(np.float32) / 255.0)
    lab_target = rgb2lab(target.astype(np.float32) / 255.0)
    ciede2000 = float(deltaE_ciede2000(lab_output, lab_target).mean())
    return {
        "mse": float(np.mean((output_float - target_float) ** 2)),
        "psnr_db": float(peak_signal_noise_ratio(target, output, data_range=255)),
        "ssim": float(structural_similarity(target, output, channel_axis=2, data_range=255)),
        "ciede2000": ciede2000,
    }


def evaluate(model_name, checkpoint_path, test_dir, output_dir, image_size, max_images,
             device, scan_mode="four_way"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pairs = resolve_pairs(test_dir)
    if max_images is not None:
        pairs = pairs[:max_images]
    net = load_generator(model_name, checkpoint_path, device, scan_mode=scan_mode)
    rows = []

    with torch.inference_mode():
        for name, input_path, target_path in pairs:
            input_image = Image.open(input_path).convert("RGB")
            target_image = Image.open(target_path).convert("RGB").resize(
                image_size, Image.Resampling.BICUBIC
            )
            prediction = tensor_to_image(
                net(image_to_tensor(input_image, image_size, device))
            )
            metrics = compute_metrics(prediction, np.asarray(target_image))
            rows.append({"image": name, **metrics})
            Image.fromarray(prediction).save(output_path / f"{Path(name).stem}_output.png")

    if not rows:
        raise RuntimeError("No image pairs were evaluated.")
    metric_names = ["mse", "psnr_db", "ssim", "ciede2000"]
    summary = {
        "model": model_name,
        "scan_mode": scan_mode,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "test_dir": str(Path(test_dir).resolve()),
        "num_images": len(rows),
        "metrics": {
            metric: {
                "mean": float(np.mean([row[metric] for row in rows])),
                "std": float(np.std([row[metric] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
            }
            for metric in metric_names
        },
    }
    with (output_path / "per_image_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", *metric_names])
        writer.writeheader()
        writer.writerows(rows)
    with (output_path / "evaluation_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    with (output_path / "evaluation_summary.txt").open("w") as handle:
        handle.write(f"Model: {model_name}\nImages: {len(rows)}\n")
        for metric in metric_names:
            values = summary["metrics"][metric]
            handle.write(f"{metric}: {values['mean']:.6f} +/- {values['std']:.6f}\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["pfan", "mamba_pfan"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--scan-mode",
        choices=["horizontal_forward", "horizontal_bidirectional",
                 "vertical_bidirectional", "four_way"],
        default="four_way",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    summary = evaluate(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        image_size=(args.image_size, args.image_size),
        max_images=args.max_images,
        device=torch.device(args.device),
        scan_mode=args.scan_mode,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
