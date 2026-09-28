"""Tests for the fusion modules, on tiny random tensors.

None of these need the dataset, a GPU or a download. The failures they pin are
the ones that otherwise cost a full training run to find.
"""
import torch

from ca_agfn.model import (
    AdaptiveGatedFusion,
    CrossModalAttention,
    SemanticClash,
    attention_entropy,
    masked_mean,
)

HIDDEN = 32


def test_attention_survives_a_fully_padded_row():
    """A meme with no text at all gives a row where every key is padding.
    Softmax over a row of -inf is NaN, and one NaN in the batch is a NaN loss,
    a NaN gradient and a dead model."""
    attention = CrossModalAttention(HIDDEN, n_heads=4)
    mask = torch.ones(3, 6, dtype=torch.long)
    mask[1] = 0

    text_out, vision_out, weights = attention(
        torch.randn(3, 6, HIDDEN), torch.randn(3, 5, HIDDEN), mask)
    assert torch.isfinite(text_out).all()
    assert torch.isfinite(vision_out).all()
    assert torch.isfinite(weights).all()


def test_attention_weights_are_a_distribution_in_eval_mode():
    attention = CrossModalAttention(HIDDEN, n_heads=4).eval()
    _, _, weights = attention(
        torch.randn(2, 6, HIDDEN), torch.randn(2, 5, HIDDEN),
        torch.ones(2, 6, dtype=torch.long))
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_cross_modal_attention_returns_per_head_weights():
    attention = CrossModalAttention(HIDDEN, n_heads=4).eval()
    _, _, weights = attention(
        torch.randn(2, 6, HIDDEN), torch.randn(2, 5, HIDDEN),
        torch.ones(2, 6, dtype=torch.long))
    assert weights.shape == (2, 4, 6, 5), "batch, heads, text tokens, patches"


def test_entropy_is_one_when_attention_is_flat():
    flat = torch.full((2, 6, 8), 1 / 8)
    assert torch.allclose(attention_entropy(flat), torch.ones(2, 1), atol=1e-4)


def test_entropy_is_zero_when_attention_is_a_spike():
    spike = torch.zeros(2, 6, 8)
    spike[..., 3] = 1.0
    assert torch.allclose(attention_entropy(spike), torch.zeros(2, 1), atol=1e-4)


def test_entropy_ignores_attention_dropout():
    """Dropout leaves weights summing to about 0.9. The gate has to mean the
    same thing in training and evaluation, so the entropy renormalises."""
    flat = torch.full((2, 6, 8), 1 / 8)
    assert torch.allclose(
        attention_entropy(flat), attention_entropy(flat * 0.9), atol=1e-4)


def test_entropy_does_not_move_when_the_patch_count_changes():
    wide = attention_entropy(torch.full((1, 4, 64), 1 / 64))
    narrow = attention_entropy(torch.full((1, 4, 8), 1 / 8))
    assert torch.allclose(wide, narrow, atol=1e-4)


def test_entropy_is_taken_per_head_not_after_averaging():
    """Eight heads, each certain about a different patch: every head has zero
    entropy, and their average is uniform. Entropy is concave, so averaging
    first is not a detail."""
    heads, patches = 8, 8
    per_head = torch.zeros(1, heads, 4, patches)
    for head in range(heads):
        per_head[:, head, :, head] = 1.0

    assert attention_entropy(per_head).item() < 0.01
    assert attention_entropy(per_head.mean(dim=1)).item() > 0.99


def test_entropy_accepts_both_shapes():
    flat3 = torch.full((2, 5, 8), 1 / 8)
    flat4 = torch.full((2, 4, 5, 8), 1 / 8)
    assert attention_entropy(flat3).shape == (2, 1)
    assert attention_entropy(flat4).shape == (2, 1)


def test_entropy_ignores_padding_positions():
    """The bug behind the first version's flat entropy.

    Two real tokens point sharply at one patch; six padding tokens attend
    evenly. Unmasked, the padding outvotes the text and the entropy looks near
    the maximum. Masked, it is the text's own entropy: zero.
    """
    weights = torch.full((1, 4, 8, 10), 1 / 10)
    weights[:, :, :2] = 0.0
    weights[:, :, :2, 7] = 1.0
    mask = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0]])

    assert attention_entropy(weights).item() > 0.7
    assert attention_entropy(weights, mask).item() < 0.01


def test_masked_mean_ignores_padding():
    states = torch.tensor([[[1.0], [3.0], [100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert masked_mean(states, mask).item() == 2.0


def test_clash_is_finite_when_the_modalities_agree_exactly():
    same = torch.randn(3, HIDDEN)
    assert torch.isfinite(SemanticClash(HIDDEN)(same, same)).all()


def test_clash_ignores_the_scale_of_its_inputs():
    """Both vectors are normalised, so the clash depends on their directions:
    the quantity CLIP aligned, not the norms it did not."""
    clash = SemanticClash(HIDDEN)
    text, image = torch.randn(3, HIDDEN), torch.randn(3, HIDDEN)
    assert torch.allclose(clash(text, image), clash(2 * text, 5 * image),
                          atol=1e-5)


def test_gate_wiring_is_monotonic_in_entropy():
    """The gate can express the design's claim, if it learns the right sign.

    The entropy weight is set by hand to the direction the architecture
    assumes, so this checks the wiring, not what training does. What training
    does is measured on the test split and reported in the README.
    """
    torch.manual_seed(0)
    fusion = AdaptiveGatedFusion(HIDDEN)
    with torch.no_grad():
        fusion.text_gate.weight[0, -1] = -4.0
        fusion.blend.fill_(4.0)

    text, image, clash = (torch.randn(4, HIDDEN) for _ in range(3))
    _, sharp = fusion(text, image, clash, torch.zeros(4, 1))
    _, flat = fusion(text, image, clash, torch.ones(4, 1))
    assert (flat < sharp).all()


def test_gate_runs_with_either_input_switched_off():
    text, image, clash = (torch.randn(4, HIDDEN) for _ in range(3))
    entropy = torch.rand(4, 1)
    for use_clash, use_entropy in ((True, False), (False, True), (False, False)):
        fusion = AdaptiveGatedFusion(HIDDEN, use_clash, use_entropy)
        fused, gate = fusion(text, image, clash if use_clash else None,
                             entropy if use_entropy else None)
        assert fused.shape == (4, HIDDEN)
        assert ((gate >= 0) & (gate <= 1)).all()
