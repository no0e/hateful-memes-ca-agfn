"""Train CA-AGFN in two phases.

    python scripts/train.py                  # the real run
    python scripts/train.py --smoke          # 64 samples, one epoch each

The smoke run exists so the whole path can be exercised on a CPU in a couple of
minutes. It proves the code runs; it proves nothing about the model.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer, CLIPImageProcessor  # noqa: E402

from ca_agfn.config import Config  # noqa: E402
from ca_agfn.data import build_loaders, write_history  # noqa: E402
from ca_agfn.model import CAAGFN  # noqa: E402
from ca_agfn.training import evaluate, train_phase  # noqa: E402


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="64 samples and one epoch per phase.")
    parser.add_argument("--out", default=str(ROOT / "docs"))
    args = parser.parse_args()

    config = Config()
    if args.data:
        config.data_dir = Path(args.data)
    if args.batch_size:
        config.batch_size = args.batch_size
    limit = 64 if args.smoke else None
    if args.smoke:
        config.phase1_epochs = 1
        config.phase2_epochs = 1
        config.batch_size = min(config.batch_size, 8)
        config.num_workers = 0

    seed_everything(config.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(config.text_model)
    image_processor = CLIPImageProcessor.from_pretrained(config.vision_model)
    train_loader, val_loader, pos_weight = build_loaders(
        config, tokenizer, image_processor, limit=limit)
    print(f"Train {len(train_loader.dataset):,}  "
          f"val {len(val_loader.dataset):,}  "
          f"pos_weight {pos_weight.item():.3f}  "
          f"captions {'yes' if train_loader.dataset.has_captions else 'no'}")

    model = CAAGFN(config)

    # Phase 1: the new modules learn against frozen backbones. Starting with
    # everything unfrozen sends a large gradient from a randomly initialised
    # head straight into pretrained weights, which is how the useful part of a
    # pretrained encoder gets destroyed in the first few steps.
    model.freeze_backbones()
    history, phase1_path, phase1_auroc = train_phase(
        model, train_loader, val_loader, config, 1, pos_weight, device,
        config.checkpoint_dir)

    # Phase 2: open the top blocks only. Embeddings and position tables stay
    # frozen; unfreezing those is what made the earlier version diverge.
    model.unfreeze_top(config.unfreeze_top)
    history, phase2_path, phase2_auroc = train_phase(
        model, train_loader, val_loader, config, 2, pos_weight, device,
        config.checkpoint_dir, history=history)

    best_path = phase2_path if phase2_auroc >= phase1_auroc else phase1_path
    best_auroc = max(phase1_auroc, phase2_auroc)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_history(history, out / "history.json")

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    final = evaluate(model, val_loader, device)

    summary = {
        "phase1_auroc": phase1_auroc,
        "phase2_auroc": phase2_auroc,
        "best_auroc": best_auroc,
        "best_checkpoint": str(best_path),
        "final": final,
        "skipped_steps_total": int(sum(history["skipped_steps"])),
        "smoke": args.smoke,
    }
    (out / "results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nPhase 1 best AUROC {phase1_auroc:.4f}")
    print(f"Phase 2 best AUROC {phase2_auroc:.4f}")
    print(f"Final: AUROC {final['auroc']:.4f}  "
          f"accuracy {final['accuracy']:.4f}  F1 {final['f1_macro']:.4f}")
    skipped = summary["skipped_steps_total"]
    print(
        f"Steps skipped as non-finite: {skipped}"
        + ("  (the gradient guard did its job)" if skipped else
           "  (nothing diverged)")
    )
    print(f"Wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
