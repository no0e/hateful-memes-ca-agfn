"""The gradient guard, and why it has to sit where it does."""
import torch

from ca_agfn.training import ExponentialMovingAverage, gradients_are_finite


def test_gradient_guard_sees_a_nan():
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
    """Why the guard runs before the clip, demonstrated rather than asserted.

    One non-finite gradient makes the total norm non-finite, and
    clip_grad_norm_ scales every other gradient by it.
    """
    layer = torch.nn.Linear(4, 2)
    layer(torch.randn(3, 4)).sum().backward()
    assert torch.isfinite(layer.bias.grad).all()

    layer.weight.grad[0, 0] = float("nan")
    torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0)
    assert not torch.isfinite(layer.bias.grad).all(), (
        "the bias gradient was finite and untouched by the NaN, and clipping "
        "spread it anyway"
    )


def test_ema_swap_restores_the_live_weights():
    layer = torch.nn.Linear(3, 1)
    ema = ExponentialMovingAverage(layer, decay=0.9)
    live = layer.weight.detach().clone()

    with torch.no_grad():
        layer.weight.add_(1.0)
    ema.update(layer)
    moved = layer.weight.detach().clone()

    backup = ema.swap_in(layer)
    assert not torch.allclose(layer.weight, moved), "shadow weights lag behind"
    assert not torch.allclose(layer.weight, live)
    ema.swap_out(layer, backup)
    assert torch.equal(layer.weight, moved)
