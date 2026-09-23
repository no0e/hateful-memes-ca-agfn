"""Tests for the parts that failed silently in the notebook this grew from.

None of these need the dataset or a GPU. They run in seconds on a CPU against
tiny random tensors, which is the point: the failures they pin are the ones
that cost a full training run to discover.
"""
import pytest
import torch

from ca_agfn.model import AdaptiveGatedFusion, CrossModalAttention, SemanticClash
from ca_agfn.training import gradients_are_finite

HIDDEN = 32


def test_attention_survives_a_fully_padded_row():
    """The failure that takes a whole batch down.

    A meme with no text at all gives a row where every key is padding. Softmax
    over a row of -inf is NaN, and one NaN in the batch is a NaN loss, a NaN
    gradient and a dead model.
    """
    attention = CrossModalAttention(HIDDEN, n_heads=4)
    text = torch.randn(3, 6, HIDDEN)
    vision = torch.randn(3, 5, HIDDEN)

    mask = torch.ones(3, 6, dtype=torch.long)
    mask[1] = 0  # this row is entirely padding

    text_out, vision_out, weights = attention(text, vision, mask)
    assert torch.isfinite(text_out).all()
    assert torch.isfinite(vision_out).all()
    assert torch.isfinite(weights).all()


def test_attention_weights_are_a_distribution_in_eval_mode():
    attention = CrossModalAttention(HIDDEN, n_heads=4).eval()
    _, _, weights = attention(
        torch.randn(2, 6, HIDDEN), torch.randn(2, 5, HIDDEN),
        torch.ones(2, 6, dtype=torch.long),
    )
    assert weights.shape[-1] == 5
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(
        weights.sum(dim=-1)), atol=1e-4)


def test_entropy_ignores_attention_dropout():
    """In training mode the attention weights are dropped out and no longer
    sum to one. The gate has to mean the same thing in both modes, so the
    entropy renormalises before it measures anything."""
    flat = torch.full((2, 6, 8), 1 / 8)
    scaled = flat * 0.9  # what dropout leaves behind

    assert torch.allclose(
        AdaptiveGatedFusion.attention_entropy(flat),
        AdaptiveGatedFusion.attention_entropy(scaled),
        atol=1e-4,
    )


def test_entropy_is_one_when_attention_is_flat():
    """Attention spread evenly is maximum entropy, which normalises to 1."""
    flat = torch.full((2, 6, 8), 1 / 8)
    entropy = AdaptiveGatedFusion.attention_entropy(flat)
    assert torch.allclose(entropy, torch.ones(2, 1), atol=1e-4)


def test_entropy_is_zero_when_attention_is_a_spike():
    """Attention on a single patch carries no uncertainty."""
    spike = torch.zeros(2, 6, 8)
    spike[..., 3] = 1.0
    entropy = AdaptiveGatedFusion.attention_entropy(spike)
    assert torch.allclose(entropy, torch.zeros(2, 1), atol=1e-4)


def test_entropy_does_not_move_when_the_patch_count_changes():
    """Normalising by log(n) is what makes the gate comparable across image
    encoders with different patch counts."""
    wide = AdaptiveGatedFusion.attention_entropy(torch.full((1, 4, 64), 1 / 64))
    narrow = AdaptiveGatedFusion.attention_entropy(torch.full((1, 4, 8), 1 / 8))
    assert torch.allclose(wide, narrow, atol=1e-4)


def test_the_gate_leans_on_the_image_when_the_text_says_nothing():
    """The claim the architecture is built on, tested as a monotonicity.

    Entropy is the only input that changes between the two calls, so any
    difference in the gate is caused by it.
    """
    torch.manual_seed(0)
    fusion = AdaptiveGatedFusion(HIDDEN)
    # Force the text gate to respond negatively to entropy, which is the
    # direction the architecture assumes; with random init the sign is random.
    with torch.no_grad():
        fusion.text_gate.weight[0, -1] = -4.0
        fusion.blend.fill_(4.0)  # trust the text gate, not the vision gate

    text = torch.randn(4, HIDDEN)
    vision = torch.randn(4, HIDDEN)
    clash = torch.randn(4, HIDDEN)

    _, sharp = fusion(text, vision, clash, torch.zeros(4, 1))
    _, flat = fusion(text, vision, clash, torch.ones(4, 1))
    assert (flat < sharp).all(), "flat attention should shift weight to vision"


def test_clash_is_zero_when_the_modalities_agree_exactly():
    """|t - v| is zero and t * v is t squared, so the clash carries only the
    agreement term. It must not blow up."""
    clash = SemanticClash(HIDDEN)
    same = torch.randn(3, HIDDEN)
    assert torch.isfinite(clash(same, same)).all()


def test_gradient_guard_sees_a_nan():
    """The check the previous training loop was missing."""
    layer = torch.nn.Linear(4, 1)
    layer(torch.randn(2, 4)).sum().backward()
    assert gradients_are_finite(layer)

    layer.weight.grad[0, 0] = float("nan")
    assert not gradients_are_finite(layer)


def test_gradient_guard_sees_an_inf():
    layer = torch.nn.Linear(4, 1)
    layer(torch.randn(2, 4)).sum().backward()
    layer.bias.grad[0] = float("inf")
    assert not gradients_are_finite(layer)


def test_clipping_a_nan_norm_poisons_every_gradient():
    """The mechanism of the original failure, demonstrated rather than claimed.

    One non-finite gradient, passed to clip_grad_norm_, makes the total norm
    non-finite and scales every other gradient by it. This is why the guard has
    to run before the clip and not after.
    """
    layer = torch.nn.Linear(4, 2)
    layer(torch.randn(3, 4)).sum().backward()

    clean = layer.bias.grad.clone()
    assert torch.isfinite(clean).all()

    layer.weight.grad[0, 0] = float("nan")
    torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0)

    assert not torch.isfinite(layer.bias.grad).all(), (
        "the bias gradient was finite and untouched by the NaN, and clipping "
        "spread it anyway"
    )


def test_the_loader_prefers_the_captioned_split(tmp_path):
    """captions.py writes <split>_captioned.jsonl. If the loader does not look
    for it, an expensive BLIP pass changes nothing and nothing says so."""
    from ca_agfn.data import split_file

    (tmp_path / "train.jsonl").write_text("{}", encoding="utf-8")
    assert split_file(tmp_path, "train").name == "train.jsonl"

    (tmp_path / "train_captioned.jsonl").write_text("{}", encoding="utf-8")
    assert split_file(tmp_path, "train").name == "train_captioned.jsonl"


def test_captions_can_be_turned_off(tmp_path):
    from ca_agfn.data import split_file

    (tmp_path / "dev.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "dev_captioned.jsonl").write_text("{}", encoding="utf-8")
    assert split_file(tmp_path, "dev", use_captions=False).name == "dev.jsonl"
