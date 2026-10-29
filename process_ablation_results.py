import zipfile
import json
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

root = Path("e:/Projects/DeSmoking")
ablation_dir = root / "ablation_results"
ablation_dir.mkdir(parents=True, exist_ok=True)

zips = [
    root / "mamba_repo/resultsofpart-1.zip",
    root / "mamba_repo/resultofpart-2.zip",
    root / "mamba_repo/resultsofpart-3.zip"
]

print("1. Extracting result archives...")
for z_path in zips:
    if not z_path.exists():
        print(f"Warning: {z_path} not found")
        continue
    with zipfile.ZipFile(z_path) as z:
        for member in z.namelist():
            # Extract eval_results, json summaries, and latest generator checkpoints
            if "eval_results/" in member or member.endswith(".json") or member.endswith("latest_net_G.pth"):
                z.extract(member, ablation_dir)
print("Extraction completed!")

# 2. Collect summaries and per-image metrics
modes = [
    ("horizontal_forward", "Horizontal (1-way Forward)"),
    ("horizontal_bidirectional", "Horizontal (2-way Bidirectional)"),
    ("vertical_bidirectional", "Vertical (2-way Bidirectional)"),
    ("four_way", "Four-Way Cross-Scan (4-way Full)")
]

summary_records = []
eval_dir = ablation_dir / "mamba_pfan/eval_results"

for raw_mode, display_name in modes:
    summary_file = eval_dir / raw_mode / "evaluation_summary.json"
    if summary_file.exists():
        data = json.loads(summary_file.read_text())
        m = data["metrics"]
        summary_records.append({
            "scan_mode": raw_mode,
            "display_name": display_name,
            "num_images": data.get("num_images", 103),
            "psnr_mean": m["psnr_db"]["mean"],
            "psnr_std": m["psnr_db"]["std"],
            "ssim_mean": m["ssim"]["mean"],
            "ssim_std": m["ssim"]["std"],
            "ciede_mean": m["ciede2000"]["mean"],
            "ciede_std": m["ciede2000"]["std"],
            "mse_mean": m["mse"]["mean"],
            "mse_std": m["mse"]["std"],
        })

# Save unified CSV
csv_path = ablation_dir / "ablation_summary.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Scan Mode", "Scanning Mechanism", "Images", "PSNR (dB)", "PSNR Std", "SSIM", "SSIM Std", "CIEDE2000", "CIEDE2000 Std", "MSE", "MSE Std"])
    for r in summary_records:
        writer.writerow([
            r["scan_mode"],
            r["display_name"],
            r["num_images"],
            f"{r['psnr_mean']:.2f}",
            f"{r['psnr_std']:.2f}",
            f"{r['ssim_mean']:.4f}",
            f"{r['ssim_std']:.4f}",
            f"{r['ciede_mean']:.2f}",
            f"{r['ciede_std']:.2f}",
            f"{r['mse_mean']:.2f}",
            f"{r['mse_std']:.2f}",
        ])
print(f"Saved {csv_path}")

# 3. Create Side-by-side Visual Comparisons for key test frames
visuals_dir = ablation_dir / "visual_comparisons"
visuals_dir.mkdir(parents=True, exist_ok=True)

dataset_zip = root / "desmoking-dataset.zip"
test_frames = ["0001", "0007", "0015", "0020"]

if dataset_zip.exists():
    with zipfile.ZipFile(dataset_zip) as dz:
        for frame_id in test_frames:
            in_name = f"test/A/{frame_id}.bmp"
            gt_name = f"test/B/{frame_id}.bmp"
            
            if in_name not in dz.namelist() or gt_name not in dz.namelist():
                continue
                
            img_in = Image.open(dz.open(in_name)).convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
            img_gt = Image.open(dz.open(gt_name)).convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
            
            variant_imgs = []
            for raw_mode, display_name in modes:
                pred_path = eval_dir / raw_mode / f"{frame_id}_output.png"
                if pred_path.exists():
                    p_img = Image.open(pred_path).convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
                else:
                    p_img = Image.new("RGB", (256, 256), (128, 128, 128))
                variant_imgs.append(p_img)
                
            # Create composite: [Hazy] [1-Way] [2-Way H] [2-Way V] [4-Way Cross] [Clean GT]
            all_cols = [img_in] + variant_imgs + [img_gt]
            col_titles = ["Input (Hazy)", "1-Way Forward", "2-Way Horiz.", "2-Way Vert.", "4-Way Cross-Scan", "Ground Truth"]
            
            w, h = 256, 256
            header_h = 36
            banner_w = w * len(all_cols)
            banner_h = h + header_h
            
            composite = Image.new("RGB", (banner_w, banner_h), (25, 25, 25))
            draw = ImageDraw.Draw(composite)
            
            for idx, (img, title) in enumerate(zip(all_cols, col_titles)):
                composite.paste(img, (idx * w, header_h))
                draw.text((idx * w + 10, 10), title, fill=(255, 255, 255))
                
            comp_path = visuals_dir / f"ablation_comparison_{frame_id}.png"
            composite.save(comp_path)
            print(f"Generated visual comparison: {comp_path}")

print("All processing successfully finished!")
