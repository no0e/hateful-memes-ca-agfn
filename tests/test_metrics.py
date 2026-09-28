"""Metrics, thresholds and the modality shuffle."""
import math

import numpy as np
import torch

from ca_agfn.evaluation import derangement
from ca_agfn.metrics import (
    auroc,
    best_threshold,
    bootstrap_auroc,
    finite_or_none,
    pearson,
)


def test_the_threshold_is_the_one_that_separates_the_classes():
    labels = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.55, 0.6, 0.65, 0.8, 0.85, 0.9])
    threshold = best_threshold(labels, probabilities)
    assert 0.65 < threshold <= 0.8, "0.5 would call every meme hateful"


def test_the_bootstrap_interval_contains_the_estimate():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 300)
    probabilities = np.clip(labels * 0.3 + rng.random(300) * 0.7, 0, 1)
    low, high = bootstrap_auroc(labels, probabilities, n=300)
    assert low < auroc(labels, probabilities) < high


def test_a_single_class_has_no_auroc():
    assert math.isnan(auroc([1, 1, 1], [0.2, 0.5, 0.9]))
    assert finite_or_none(float("nan")) is None


def test_pearson_is_undefined_for_a_constant():
    assert math.isnan(pearson([1, 1, 1], [0.1, 0.5, 0.9]))


def test_the_shuffle_moves_every_meme():
    """A shuffled meme that keeps its own image measures nothing."""
    generator = torch.Generator().manual_seed(0)
    for n in (2, 3, 32):
        order = derangement(n, generator)
        assert sorted(order.tolist()) == list(range(n))
        assert (order != torch.arange(n)).all()
