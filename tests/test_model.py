"""The whole model and the whole pipeline, on a tiny random CLIP.

These download a few megabytes once and then run on a CPU in seconds.
"""
import importlib.util
import json

import pytest
import torch
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

from ca_agfn.checkpoint import load_checkpoint, save_checkpoint
from ca_agfn.config import ROOT, VARIANTS, config_for
from ca_agfn.model import CAAGFN
from ca_agfn.synthetic import write_tiny_text_model
from ca_agfn.training import parameter_groups

pytestmark = pytest.mark.network
HIDDEN = 64  # the tiny CLIP's projection width


def tiny_config(tiny_clip, variant="full", **overrides):
    settings = {"text_model": tiny_clip, "vision_model": tiny_clip,
                "hidden_size": HIDDEN, **overrides}
    return config_for(variant, **settings)


def inputs(tiny_clip, batch=3):
    tokenizer = AutoTokenizer.from_pretrained(tiny_clip)
    encoded = tokenizer(["a cat", "no need to panic at all", ""][:batch],
                        padding="max_length", max_length=12,
                        truncation=True, return_tensors="pt")
    size = CLIPImageProcessor.from_pretrained(tiny_clip).crop_size["height"]
    pixels = torch.randn(batch, 3, size, size)
    return encoded["input_ids"], encoded["attention_mask"], pixels


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_variant_trains(tiny_clip, tmp_path, variant):
    overrides = {}
    if "text_model" in VARIANTS[variant]:
        overrides["text_model"] = str(
            write_tiny_text_model(tmp_path / "text", tiny_clip))
    model = CAAGFN(tiny_config(tiny_clip, variant, **overrides))
    model.unfreeze_top(2)

    logits = model(*inputs(tiny_clip))
    assert logits.shape == (3,)
    logits.sum().backward()
    assert any(p.grad is not None for p in model.head.parameters())


def test_the_projections_start_as_clips_own(tiny_clip):
    """At initialisation the pooled vectors are exactly CLIP's aligned text and
    image embeddings, so the clash starts out comparing like with like."""
    model = CAAGFN(tiny_config(tiny_clip)).eval()
    input_ids, mask, pixels = inputs(tiny_clip)

    with torch.no_grad():
        _, text = model.encode_text(input_ids, mask)
        _, image = model.encode_image(pixels)
        clip_text = CLIPTextModelWithProjection.from_pretrained(tiny_clip)(
            input_ids=input_ids, attention_mask=mask).text_embeds
        clip_image = CLIPVisionModelWithProjection.from_pretrained(tiny_clip)(
            pixel_values=pixels).image_embeds

    assert torch.allclose(text, clip_text, atol=1e-5)
    assert torch.allclose(image, clip_image, atol=1e-5)


def test_the_classifier_reads_the_clash(tiny_clip):
    """The clash reaches the classifier directly, not only through the gate."""
    model = CAAGFN(tiny_config(tiny_clip)).eval()
    seen = {}
    model.head.register_forward_hook(
        lambda module, args, output: seen.update(features=args[0]))
    model.clash.register_forward_hook(
        lambda module, args, output: seen.update(clash=output))

    with torch.no_grad():
        model(*inputs(tiny_clip))
    assert seen["features"].shape[-1] == 2 * HIDDEN
    assert torch.equal(seen["features"][:, HIDDEN:], seen["clash"])


def test_only_the_top_blocks_open(tiny_clip):
    model = CAAGFN(tiny_config(tiny_clip))
    model.unfreeze_top(2)

    for name, parameter in model.named_parameters():
        if model.is_custom(name):
            assert parameter.requires_grad, name
        elif ".layers.3." in name or ".layers.4." in name:
            assert parameter.requires_grad, name
        else:
            assert not parameter.requires_grad, name


def test_deeper_blocks_get_smaller_learning_rates(tiny_clip):
    config = tiny_config(tiny_clip)
    model = CAAGFN(config)
    model.unfreeze_top(2)
    rates = {group["name"]: group["lr"]
             for group in parameter_groups(model, config, phase=2)}

    assert rates["backbone_depth_0"] == config.phase2_lr_backbone
    assert rates["backbone_depth_1"] == pytest.approx(
        config.phase2_lr_backbone * config.llrd_decay)
    assert rates["head"] == config.phase2_lr_head


def test_a_checkpoint_round_trips(tiny_clip, tmp_path):
    model = CAAGFN(tiny_config(tiny_clip)).eval()
    with torch.no_grad():
        for parameter in model.head.parameters():
            parameter.add_(0.5)  # so a fresh model would disagree
    batch = inputs(tiny_clip)

    path = save_checkpoint(model, tmp_path / "model.safetensors")
    restored, meta = load_checkpoint(path)
    with torch.no_grad():
        assert torch.allclose(model(*batch), restored(*batch), atol=1e-6)
    assert meta["config"]["hidden_size"] == HIDDEN


def _train_script():
    spec = importlib.util.spec_from_file_location(
        "train_script", ROOT / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_smoke_run_writes_a_complete_result(tiny_clip, tmp_path):
    results = _train_script().main(
        ["--smoke", "--results", str(tmp_path), "--seed", "0"])
    written = json.loads(
        (tmp_path / "full" / "seed0.json").read_text(encoding="utf-8"))

    assert written["variant"] == "full" and written["smoke"]
    for key in ("auroc", "auroc_ci95", "accuracy", "f1_macro", "threshold",
                "auroc_shuffled_image", "auroc_shuffled_text"):
        assert key in written["test"], key
    assert len(written["predictions"]["probability"]) == 32
    assert "label" not in written["predictions"], "labels are Meta's to share"
    assert results["training"]["skipped_steps"] == 0


def test_a_text_only_model_is_blind_to_the_image_shuffle(tiny_clip, tmp_path):
    """A sanity check on the shuffle test itself: moving images between memes
    cannot change what a model that never reads images predicts."""
    results = _train_script().main(
        ["--smoke", "--variant", "text_only", "--results", str(tmp_path)])
    test = results["test"]
    assert test["auroc_shuffled_image"] == pytest.approx(test["auroc"])
