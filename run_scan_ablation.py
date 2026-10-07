"""Run or print the Cholec80 Mamba scan-direction ablation matrix.

Use --dry-run locally. Run without --dry-run on Kaggle after setting --dataroot
and --checkpoints-dir to persistent storage.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCAN_MODES = (
    "horizontal_forward",
    "horizontal_bidirectional",
    "vertical_bidirectional",
    "four_way",
)


def build_command(args, mode):
    command = [
        sys.executable,
        str(Path(__file__).with_name("train.py")),
        "--dataroot", args.dataroot,
        "--name", f"{args.name}_{mode}",
        "--model", "pix2pix",
        "--netG", "mamba_pfan",
        "--netD", "basic",
        "--direction", "AtoB",
        "--dataset_mode", "aligned",
        "--norm", "batch",
        "--checkpoints_dir", args.checkpoints_dir,
        "--embed_dim", str(args.embed_dim),
        "--ndf", str(args.ndf),
        "--ngf", str(args.ngf),
        "--batch_size", str(args.batch_size),
        "--num_threads", str(args.num_threads),
        "--n_epochs", str(args.n_epochs),
        "--n_epochs_decay", str(args.n_epochs_decay),
        "--save_epoch_freq", str(args.save_epoch_freq),
        "--display_id", "-1",
        "--scan_mode", mode,
    ]
    if args.gpu_ids is not None:
        command.extend(["--gpu_ids", args.gpu_ids])
    if args.no_flip:
        command.append("--no_flip")
    return command


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--modes", nargs="+", choices=SCAN_MODES, default=list(SCAN_MODES))
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest-output", default="scan_ablation_commands.json")
    args = parser.parse_args()

    commands = [build_command(args, mode) for mode in args.modes]
    manifest = {
        "dataset": "Cholec80",
        "model": "mamba_pfan",
        "modes": args.modes,
        "commands": [command for command in commands],
        "shared_settings": {
            "ngf": args.ngf,
            "ndf": args.ndf,
            "embed_dim": args.embed_dim,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "n_epochs_decay": args.n_epochs_decay,
            "no_flip": args.no_flip,
        },
    }
    Path(args.manifest_output).write_text(json.dumps(manifest, indent=2))

    for command in commands:
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=Path(__file__).parent)


if __name__ == "__main__":
    main()
