"""Train one variant with one seed, then score it once on the test split.

    python scripts/train.py                            # the full model
    python scripts/train.py --variant concat --seed 1  # one ablation
    python scripts/train.py --smoke                    # fake data, tiny model

Writes results/<variant>/seed<seed>.json: the config, the training history,
validation and test scores with a bootstrap interval, the AUROC with images and
with texts shuffled between memes, the gate statistics, and per-meme outputs
for the figures. Nothing in it comes from the dataset but ids.

The smoke run generates a fake dataset and uses a tiny random CLIP, so it runs
on a CPU in under a minute with nothing downloaded but a few megabytes. It
proves the pipeline runs end to end. It proves nothing about the model.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, CLIPImageProcessor
from transformers.utils import logging as hf_logging

from ca_agfn.checkpoint import save_checkpoint
from ca_agfn.config import ROOT, VARIANTS, config_for
from ca_agfn.data import build_loaders
from ca_agfn.evaluation import report
from ca_agfn.metrics import finite_or_none
from ca_agfn.model import CAAGFN
from ca_agfn.synthetic import write_fake_dataset, write_tiny_text_model
from ca_agfn.training import fit, seed_everything

TINY_CLIP = "hf-internal-testing/tiny-random-CLIPModel"


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def clean(value):
    """Recursively swap NaN for None, which JSON can hold."""
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return finite_or_none(value)


def parse(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="full", choices=sorted(VARIANTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--phase1-epochs", type=int, default=None)
    parser.add_argument("--phase2-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--unfreeze", type=int, default=None,
                        help="How many top blocks of each backbone to open.")
    parser.add_argument("--results", default=str(ROOT / "results"),
                        help="Directory for <variant>/seed<seed>.json.")
    parser.add_argument("--checkpoint", default=None,
                        help="Also save the tuned weights here (.safetensors).")
    parser.add_argument("--smoke", action="store_true",
                        help="Fake data and a tiny model, one epoch per phase.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    hf_logging.set_verbosity_error()

    overrides = {"seed": args.seed}
    for key, value in (("data_dir", args.data), ("batch_size", args.batch_size),
                       ("num_workers", args.workers),
                       ("phase1_epochs", args.phase1_epochs),
                       ("phase2_epochs", args.phase2_epochs),
                       ("patience", args.patience),
                       ("unfreeze_top", args.unfreeze)):
        if value is not None:
            overrides[key] = value

    scratch = None
    if args.smoke:
        scratch = tempfile.TemporaryDirectory()
        overrides.update({
            "data_dir": write_fake_dataset(Path(scratch.name) / "data"),
            "text_model": TINY_CLIP, "vision_model": TINY_CLIP,
            "hidden_size": 64, "phase1_epochs": 1, "phase2_epochs": 1,
            "batch_size": 8, "num_workers": 0, "val_size": 16,
        })
        if "text_model" in VARIANTS[args.variant]:
            overrides["text_model"] = str(write_tiny_text_model(
                Path(scratch.name) / "text_model", TINY_CLIP))
    config = config_for(args.variant, **overrides)

    seed_everything(config.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    device_name = (torch.cuda.get_device_name(0) if device.type == "cuda"
                   else "cpu")
    print(f"Variant {args.variant}, seed {config.seed}, on {device_name}")

    tokenizer = AutoTokenizer.from_pretrained(config.text_model)
    image_processor = CLIPImageProcessor.from_pretrained(config.vision_model)
    loaders, pos_weight, used_captions = build_loaders(
        config, tokenizer, image_processor, pin_memory=device.type == "cuda")
    print(f"Train {len(loaders['train'].dataset):,}  "
          f"val {len(loaders['val'].dataset):,}  "
          f"test {len(loaders['test'].dataset):,}  "
          f"pos_weight {pos_weight.item():.3f}  "
          f"captions {'yes' if used_captions else 'no'}")

    started = time.time()
    model = CAAGFN(config).to(device)
    history, summary = fit(model, loaders, config, pos_weight, device)
    scores = report(model, loaders, device, seed=config.seed)
    minutes = (time.time() - started) / 60

    test = scores["test"]
    low, high = test["auroc_ci95"]
    print(f"\nTest AUROC {test['auroc']:.4f}  [{low:.3f}, {high:.3f}]  "
          f"accuracy {test['accuracy']:.4f}  F1 {test['f1_macro']:.4f}")
    print(f"  with images shuffled {test['auroc_shuffled_image']:.4f}, "
          f"with texts shuffled {test['auroc_shuffled_text']:.4f}")
    if "gate" in scores:
        gate = scores["gate"]
        print(f"  entropy {gate['entropy_mean']:.3f} ± {gate['entropy_std']:.3f}"
              f", gate against entropy r = {gate['r_entropy']:+.3f}")
    print(f"  steps skipped as non-finite: {summary['skipped_steps']}")

    results = clean({
        "variant": args.variant,
        "seed": config.seed,
        "smoke": args.smoke,
        "commit": git_commit(),
        "device": device_name,
        "minutes": round(minutes, 1),
        "captions": used_captions,
        "config": config.to_dict(),
        "training": summary,
        "history": history,
        **scores,
    })
    out = Path(args.results) / args.variant / f"seed{config.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"Wrote {out}")

    if args.checkpoint:
        path = save_checkpoint(model, args.checkpoint, {
            "variant": args.variant, "test_auroc": test["auroc"]})
        print(f"Wrote {path}")

    if scratch is not None:
        scratch.cleanup()
    return results


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
