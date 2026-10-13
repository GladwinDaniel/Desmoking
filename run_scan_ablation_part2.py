"""Ablation Part 2/3 — Cholec80 Mamba scan-direction ablation.

Runs:
  - horizontal_bidirectional
  - vertical_bidirectional

After training completes the results are saved to a JSON file and
auto-downloaded (Kaggle / Colab compatible).

Use --dry-run locally to test without actually training.
"""

import argparse
import json
import subprocess
import sys
import shutil
from pathlib import Path
from datetime import datetime

# -- Modes handled by this file -----------------------------------------------
SCAN_MODES = (
    "horizontal_bidirectional",
    "vertical_bidirectional",
)
# -----------------------------------------------------------------------------


def build_command(args, mode):
    command = [
        sys.executable,
        str(Path(__file__).with_name("train.py")),
        "--dataroot", str(args.dataroot),
        "--name", f"{args.name}_{mode}",
        "--model", "pix2pix",
        "--netG", "mamba_pfan",
        "--netD", "basic",
        "--direction", "AtoB",
        "--dataset_mode", "aligned",
        "--norm", "batch",
        "--checkpoints_dir", str(args.checkpoints_dir),
        "--embed_dim", str(args.embed_dim),
        "--ndf", str(args.ndf),
        "--ngf", str(args.ngf),
        "--batch_size", str(args.batch_size),
        "--num_threads", str(args.num_threads),
        "--n_epochs", str(args.n_epochs),
        "--n_epochs_decay", str(args.n_epochs_decay),
        "--save_epoch_freq", str(args.save_epoch_freq),
        "--display_id", "-1",
        "--print_freq", str(args.print_freq),
        "--scan_mode", mode,
    ]
    if args.gpu_ids is not None:
        command.extend(["--gpu_ids", str(args.gpu_ids)])
    if args.no_flip:
        command.append("--no_flip")
    return command


def collect_loss_log(checkpoints_dir: Path, name: str) -> list:
    """Parse the loss_log.txt produced by pix2pix into a list of dicts."""
    log_path = checkpoints_dir / name / "loss_log.txt"
    entries = []
    if not log_path.is_file():
        return entries
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("="):
            continue
        try:
            meta_part, metrics_part = line.split(")", 1)
            meta_raw = meta_part.lstrip("(")
            meta = dict(
                kv.strip().split(": ", 1)
                for kv in meta_raw.split(",")
                if ": " in kv
            )
            metrics = dict(
                kv.strip().split(": ", 1)
                for kv in metrics_part.split(",")
                if ": " in kv
            )
            entries.append({**meta, **metrics})
        except Exception:
            pass
    return entries


def save_and_download_results(args, modes_run, dry_run):
    """Save results to JSON and trigger a download in notebook environments."""
    checkpoints_dir = Path(args.checkpoints_dir)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_filename = f"ablation_part2_results_{timestamp}.json"

    results = {
        "part": 2,
        "modes_run": modes_run,
        "timestamp_utc": timestamp,
        "dry_run": dry_run,
        "shared_settings": {
            "ngf": args.ngf,
            "ndf": args.ndf,
            "embed_dim": args.embed_dim,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "n_epochs_decay": args.n_epochs_decay,
            "print_freq": args.print_freq,
            "no_flip": args.no_flip,
        },
        "per_mode_results": {},
    }

    for mode in modes_run:
        run_name = f"{args.name}_{mode}"
        ckpt_path = checkpoints_dir / run_name / "latest_net_G.pth"
        results["per_mode_results"][mode] = {
            "checkpoint_exists": ckpt_path.is_file(),
            "checkpoint_path": str(ckpt_path),
            "loss_log": collect_loss_log(checkpoints_dir, run_name),
        }

    output_path = Path(output_filename)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n[RESULTS] Saved -> {output_path.resolve()}", flush=True)

    # Kaggle
    try:
        from kaggle_secrets import UserSecretsClient  # noqa: F401
        shutil.copy(output_path, f"/kaggle/working/{output_filename}")
        print(f"[DOWNLOAD] Copied to /kaggle/working/{output_filename} — visible in Kaggle output panel.", flush=True)
    except ImportError:
        pass

    # Google Colab
    try:
        from google.colab import files as colab_files
        colab_files.download(str(output_path))
        print(f"[DOWNLOAD] Colab download triggered for {output_filename}.", flush=True)
    except (ImportError, Exception):
        pass

    print(f"[DOWNLOAD] Manual fallback path:\n  {output_path.resolve()}", flush=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Ablation Part 2/3 -- horizontal_bidirectional + vertical_bidirectional"
    )
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--checkpoints-dir", default="./checkpoints")
    parser.add_argument("--name", default="mamba_scan")
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--n-epochs-decay", type=int, default=0)
    parser.add_argument("--save-epoch-freq", type=int, default=5)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest-output", default="scan_ablation_part2_commands.json")
    args = parser.parse_args()

    print(f"[PART 2/3] Modes to run: {list(SCAN_MODES)}", flush=True)

    commands = [build_command(args, mode) for mode in SCAN_MODES]
    manifest = {
        "part": 2,
        "dataset": "Cholec80",
        "model": "mamba_pfan",
        "modes": list(SCAN_MODES),
        "commands": commands,
        "shared_settings": {
            "ngf": args.ngf,
            "ndf": args.ndf,
            "embed_dim": args.embed_dim,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "n_epochs_decay": args.n_epochs_decay,
            "print_freq": args.print_freq,
            "no_flip": args.no_flip,
        },
    }
    Path(args.manifest_output).write_text(json.dumps(manifest, indent=2))

    modes_run = []
    checkpoints_dir = Path(args.checkpoints_dir)

    for mode in SCAN_MODES:
        ckpt = checkpoints_dir / f"{args.name}_{mode}" / "latest_net_G.pth"
        if args.skip_existing and ckpt.is_file():
            print(f"[SKIP] Existing checkpoint found for {mode}: {ckpt}", flush=True)
            modes_run.append(mode)
            continue
        command = build_command(args, mode)
        print(f"\n[RUN] {mode}\n  " + " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=Path(__file__).parent)
        modes_run.append(mode)

    print("\n[PART 2/3] All assigned modes completed.", flush=True)
    save_and_download_results(args, modes_run, args.dry_run)


if __name__ == "__main__":
    main()
