"""Reading the per-run result files back, for the table and the figure."""
import json
from pathlib import Path

import numpy as np

LABELS = {
    "full": "CA-AGFN, full",
    "no_entropy": "without the entropy input to the gate",
    "no_clash": "without the clash vector",
    "no_captions": "without BLIP captions",
    "xlmr": "XLM-R text encoder (the v1 backbone)",
    "concat": "Concatenated CLIP vectors, no fusion",
    "text_only": "Text only",
    "image_only": "Image only",
}


def load(results):
    runs = {}
    for path in sorted(Path(results).glob("*/seed*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if not run.get("smoke"):
            runs.setdefault(run["variant"], []).append(run)
    return runs


def spread(values):
    values = np.array([v for v in values if v is not None], dtype=float)
    if not len(values):
        return None
    return {"mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "n": int(len(values))}


def summarise(runs):
    summary = {}
    for variant, group in runs.items():
        test = [run["test"] for run in group]
        gate = [run.get("gate") for run in group if run.get("gate")]
        summary[variant] = {
            "label": LABELS.get(variant, variant),
            "seeds": sorted(run["seed"] for run in group),
            "test_auroc": spread(t["auroc"] for t in test),
            "test_accuracy": spread(t["accuracy"] for t in test),
            "test_f1_macro": spread(t["f1_macro"] for t in test),
            "auroc_shuffled_image": spread(
                t["auroc_shuffled_image"] for t in test),
            "auroc_shuffled_text": spread(
                t["auroc_shuffled_text"] for t in test),
            "val_auroc": spread(run["val"]["auroc"] for run in group),
            "r_gate_entropy": spread(g["r_entropy"] for g in gate),
            "entropy_std": spread(g["entropy_std"] for g in gate),
            "minutes": spread(run["minutes"] for run in group),
            "skipped_steps": int(sum(run["training"]["skipped_steps"]
                                     for run in group)),
        }
    return summary


def cell(stat, digits=3):
    if stat is None:
        return "–"
    if stat["n"] == 1:
        return f"{stat['mean']:.{digits}f}"
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def table(summary):
    lines = [
        "| Model | Seeds | Test AUROC | Accuracy | Macro F1 "
        "| AUROC, images shuffled | AUROC, texts shuffled |",
        "|---|---|---|---|---|---|---|",
    ]
    for variant in [v for v in LABELS if v in summary] + sorted(
            set(summary) - set(LABELS)):
        row = summary[variant]
        lines.append(
            f"| {row['label']} | {len(row['seeds'])} "
            f"| {cell(row['test_auroc'])} | {cell(row['test_accuracy'])} "
            f"| {cell(row['test_f1_macro'])} "
            f"| {cell(row['auroc_shuffled_image'])} "
            f"| {cell(row['auroc_shuffled_text'])} |")
    return "\n".join(lines)
