"""Scoring a trained model, once, on the test split.

Besides the usual numbers this runs the check the architecture's premise
invites: if a meme is hateful in the gap between its text and its image, then
pairing each text with somebody else's image should hurt. `evaluate` with
`shuffle="image"` does exactly that, inside each batch, and keeps the label with
the text. The drop in AUROC is how much the model actually uses the image. The
same with `shuffle="text"` measures its use of the text.
"""
import numpy as np
import torch
from sklearn.metrics import roc_curve

from .metrics import (
    auroc,
    best_threshold,
    bootstrap_auroc,
    classification,
    pearson,
)


def derangement(n, generator):
    """A permutation with no fixed points, so every meme gets another's input."""
    order = torch.randperm(n, generator=generator)
    shifted = torch.empty_like(order)
    shifted[order] = order.roll(1)
    return shifted


@torch.no_grad()
def evaluate(model, loader, device, shuffle=None, seed=0):
    """Probabilities for every meme in `loader`, in order.

    Also the gate and the entropy, when the model has them. `shuffle` is None,
    "image" or "text".
    """
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    collected = {"probability": [], "label": [], "gate": [], "entropy": []}

    for batch in loader:
        input_ids = batch["input_ids"]
        mask = batch["attention_mask"]
        pixels = batch["pixel_values"]
        if shuffle is not None and len(input_ids) > 1:
            order = derangement(len(input_ids), generator)
            if shuffle == "image":
                pixels = pixels[order]
            elif shuffle == "text":
                input_ids, mask = input_ids[order], mask[order]
            else:
                raise ValueError(f"Cannot shuffle {shuffle!r}.")

        logits, details = model(input_ids.to(device), mask.to(device),
                                pixels.to(device), return_details=True)
        collected["probability"].append(torch.sigmoid(logits.float()).cpu())
        collected["label"].append(batch["label"])
        for key in ("gate", "entropy"):
            if key in details:
                collected[key].append(details[key].float().cpu())

    return {key: torch.cat(values).numpy()
            for key, values in collected.items() if values}


def report(model, loaders, device, seed=0):
    """Everything the README reports about one trained run."""
    val = evaluate(model, loaders["val"], device)
    test = evaluate(model, loaders["test"], device)
    labels, probabilities = test["label"], test["probability"]

    threshold = best_threshold(val["label"], val["probability"])
    result = {
        "val": {"auroc": auroc(val["label"], val["probability"])},
        "test": {
            "auroc": auroc(labels, probabilities),
            "auroc_ci95": bootstrap_auroc(labels, probabilities, seed=seed),
            "threshold": threshold,
            **classification(labels, probabilities, threshold),
        },
    }

    for modality in ("image", "text"):
        shuffled = evaluate(model, loaders["test"], device, shuffle=modality,
                            seed=seed)
        result["test"][f"auroc_shuffled_{modality}"] = auroc(
            shuffled["label"], shuffled["probability"])

    if "gate" in test:
        gate, entropy = test["gate"], test["entropy"]
        result["gate"] = {
            "r_entropy": pearson(entropy, gate),
            "entropy_mean": float(entropy.mean()),
            "entropy_std": float(entropy.std()),
            "gate_mean": float(gate.mean()),
            "gate_std": float(gate.std()),
            "gate_mean_hateful": float(gate[labels == 1].mean()),
            "gate_mean_benign": float(gate[labels == 0].mean()),
        }

    # Per-meme outputs, so the figures can be redrawn without a GPU. No labels
    # and no text: the dataset's licence does not allow redistributing it, and
    # the ROC curve is stored as a curve instead.
    false_positive, true_positive, _ = roc_curve(labels, probabilities)
    result["roc"] = {"fpr": np.round(false_positive, 4).tolist(),
                     "tpr": np.round(true_positive, 4).tolist()}
    result["predictions"] = {
        "id": loaders["test"].dataset.ids,
        **{key: np.round(test[key], 5).tolist()
           for key in ("probability", "gate", "entropy") if key in test},
    }
    return result
