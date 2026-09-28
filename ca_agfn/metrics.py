"""Scores, with the uncertainty that 500 memes carry.

On a 500-meme split the 95% interval on an AUROC near 0.7 is about 0.05 wide
either side. Differences smaller than that between two single runs are not
findings, which is why every number in the README is a mean over seeds with
its spread, and the test AUROC of each run carries a bootstrap interval.
"""
import math

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def auroc(labels, probabilities):
    labels = np.asarray(labels)
    if labels.min() == labels.max():  # one class only
        return float("nan")
    return float(roc_auc_score(labels, probabilities))


def bootstrap_auroc(labels, probabilities, n=2000, seed=0, level=0.95):
    """Percentile bootstrap interval for the AUROC."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        index = rng.integers(0, len(labels), len(labels))
        if labels[index].min() == labels[index].max():
            continue
        scores.append(roc_auc_score(labels[index], probabilities[index]))
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(scores, [tail, 100 - tail])
    return float(low), float(high)


def best_threshold(labels, probabilities):
    """The cut that maximises accuracy, chosen on the validation hold-out.

    The loss is weighted towards positives, so the model's probabilities are
    shifted up and 0.5 is not a neutral threshold. Picking it on the test split
    would be the same leak as picking the checkpoint there.
    """
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    candidates = np.unique(np.concatenate([[0.5], probabilities]))
    accuracy = [((probabilities >= t) == labels).mean() for t in candidates]
    return float(candidates[int(np.argmax(accuracy))])


def classification(labels, probabilities, threshold):
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro")),
    }


def pearson(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def finite_or_none(value):
    """JSON has no NaN; a missing measurement is written as null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
