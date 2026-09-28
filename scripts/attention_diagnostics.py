"""Why the text's attention over the image is flat: measure it, don't guess.

    python scripts/attention_diagnostics.py --data ~/work/memes-data

Runs the trained model's text-to-image attention on the test split by hand and
reports, over real text tokens only:

  the spread of the attention logits across the 50 image positions, which is
  what the softmax turns into a distribution; a spread near zero is a uniform
  distribution whatever the queries are

  the cosine similarity between the keys of different image positions; if the
  keys are nearly identical, every query scores them alike

  the same similarity for CLIP's own patch tokens, before any trained layer,
  to see whether the sameness comes from CLIP or from the projection

  the entropy, per head, as the gate sees it

Each is measured twice: on the trained model, and on the same architecture
freshly initialised, to tell what training did from what it started with.

Writes results/attention_diagnostics.json.
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPImageProcessor
from transformers.utils import logging as hf_logging

from ca_agfn.checkpoint import load_checkpoint
from ca_agfn.config import ROOT
from ca_agfn.data import build_loaders
from ca_agfn.model import CAAGFN, attention_entropy
from ca_agfn.training import seed_everything


def mean_pairwise_cosine(vectors):
    """Mean cosine similarity over distinct pairs, per item: (..., n, d)."""
    unit = F.normalize(vectors, dim=-1)
    similarity = unit @ unit.transpose(-1, -2)
    n = similarity.size(-1)
    off_diagonal = similarity.sum(dim=(-1, -2)) - similarity.diagonal(
        dim1=-2, dim2=-1).sum(-1)
    return off_diagonal / (n * (n - 1))


@torch.no_grad()
def measure(model, loader, device, batches):
    attention = model.cross_attention.text_to_vision
    width, heads = attention.embed_dim, attention.num_heads
    size = width // heads
    w_q, w_k, _ = attention.in_proj_weight.split(width)
    b_q, b_k, _ = attention.in_proj_bias.split(width)

    totals = {"logit_std": [], "key_cosine": [], "patch_cosine": [],
              "entropy": [], "entropy_per_head_min": [], "real_tokens": []}
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        pixels = batch["pixel_values"].to(device)

        text, _ = model.encode_text(ids, mask)
        image, _ = model.encode_image(pixels)
        raw = model.vision_encoder(pixel_values=pixels).last_hidden_state

        queries = (text @ w_q.T + b_q).unflatten(-1, (heads, size)).transpose(1, 2)
        keys = (image @ w_k.T + b_k).unflatten(-1, (heads, size)).transpose(1, 2)
        logits = queries @ keys.transpose(-1, -2) / math.sqrt(size)

        real = mask.bool()
        spread = logits.std(dim=-1)  # (batch, heads, tokens)
        per_token = spread.mean(dim=1)
        totals["logit_std"].append(per_token[real])
        totals["key_cosine"].append(mean_pairwise_cosine(keys[:, :, 1:]).mean(1))
        totals["patch_cosine"].append(mean_pairwise_cosine(raw[:, 1:]))

        weights = logits.softmax(dim=-1)
        entropy = attention_entropy(weights, mask).squeeze(-1)
        if index == 0:
            # The attention above is recomputed by hand; it has to be the one
            # the model runs, or none of these numbers describe it.
            _, details = model(ids, mask, pixels, return_details=True)
            if not torch.allclose(entropy, details["entropy"], atol=1e-4):
                raise RuntimeError("Recomputed attention does not match the "
                                   "model's own.")
        totals["entropy"].append(entropy)
        per_head = torch.stack([attention_entropy(weights[:, h], mask)
                                for h in range(heads)], dim=-1).squeeze(1)
        totals["entropy_per_head_min"].append(per_head.min(dim=-1).values)
        totals["real_tokens"].append(mask.sum(dim=1).float())

    return {key: torch.cat(values).float().cpu() for key, values in totals.items()}


def describe(values):
    return {"mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max())}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=str(
        ROOT / "weights" / "ca_agfn.safetensors"))
    parser.add_argument("--data", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--out", default=str(
        ROOT / "results" / "attention_diagnostics.json"))
    args = parser.parse_args()
    hf_logging.set_verbosity_error()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, meta = load_checkpoint(args.checkpoint, device)
    config = model.config
    if args.data:
        config.data_dir = Path(args.data)
    config.num_workers = 4

    loaders, _, _ = build_loaders(
        config, AutoTokenizer.from_pretrained(config.text_model),
        CLIPImageProcessor.from_pretrained(config.vision_model))
    seed_everything(config.seed)
    untrained = CAAGFN(config).to(device).eval()

    report = {"checkpoint": Path(args.checkpoint).name}
    for name, candidate in (("trained", model), ("untrained", untrained)):
        stats = measure(candidate, loaders["test"], device, args.batches)
        report["memes"] = int(len(stats["entropy"]))
        report[name] = {key: describe(values) for key, values in stats.items()}
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{report['memes']} test memes, real text tokens only\n")
    print(f"  {'':<52} {'trained':>17}   {'untrained':>17}")
    for key, label in (
            ("logit_std", "spread of attention logits across image positions"),
            ("key_cosine", "cosine between keys of different patches"),
            ("patch_cosine", "cosine between CLIP's own patch tokens"),
            ("entropy", "entropy the gate reads"),
            ("entropy_per_head_min", "entropy of the sharpest head")):
        cells = [f"{report[n][key]['mean']:.4f} ± {report[n][key]['std']:.4f}"
                 for n in ("trained", "untrained")]
        print(f"  {label:<52} {cells[0]:>17}   {cells[1]:>17}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
