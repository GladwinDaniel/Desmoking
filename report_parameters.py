"""Report generator parameter counts for reproducible model comparisons."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.networks import define_G


def load_state_dict(path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return {
        key.removeprefix("module."): value for key, value in checkpoint.items()
    }


def report_model(model_name, ngf, checkpoint=None):
    model = define_G(
        opt=None,
        input_nc=3,
        output_nc=3,
        ngf=ngf,
        netG=model_name,
        norm="batch",
        init_type="normal",
        gpu_ids=[],
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    result = {
        "model": model_name,
        "ngf": ngf,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "total_millions": total / 1e6,
        "trainable_millions": trainable / 1e6,
    }
    if checkpoint:
        state_dict = load_state_dict(checkpoint, torch.device("cpu"))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        result["checkpoint"] = str(Path(checkpoint).resolve())
        result["checkpoint_parameters"] = sum(value.numel() for value in state_dict.values())
        result["missing_keys"] = missing
        result["unexpected_keys"] = unexpected
        result["checkpoint_compatible"] = not missing and not unexpected
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--pfan-checkpoint")
    parser.add_argument("--mamba-checkpoint")
    parser.add_argument("--published-pfan-parameters", type=int, default=629000)
    parser.add_argument("--output", default="parameter_report.json")
    args = parser.parse_args()

    results = [
        report_model("pfan", args.ngf, args.pfan_checkpoint),
        report_model("mamba_pfan", args.ngf, args.mamba_checkpoint),
    ]
    report = {
        "configuration": {
            "input_channels": 3,
            "output_channels": 3,
            "ngf": args.ngf,
            "normalization": "batch",
            "count_scope": "generator_only",
        },
        "models": results,
        "published_pfan_parameters": args.published_pfan_parameters,
        "published_to_mamba_ratio": 629000 / results[1]["total_parameters"],
        "local_pfan_to_mamba_ratio": results[0]["total_parameters"] / results[1]["total_parameters"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
