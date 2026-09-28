"""Two-phase training, and the guard that makes phase two survive.

Phase two opens the top blocks of two pretrained backbones, which is where a
non-finite gradient is most likely to turn up. The guard for it has to sit
between the backward pass and the clip.

Watching the loss instead does not work. `clip_grad_norm_` computes one total
norm across every parameter and scales them all by it, so a single non-finite
entry anywhere makes that norm non-finite and turns every other gradient into
NaN, including the ones that were fine. The optimiser writes those into the
weights, and only the *next* forward pass returns a non-finite loss. By then
the model is already gone.

So gradients are tested for finiteness after backward and before the clip. A
bad batch is dropped and the weights are never touched. `skipped_steps` counts
how often that happens, because a guard that silently eats half the batches is
its own kind of failure.
"""
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup

from .evaluation import evaluate
from .metrics import auroc


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ExponentialMovingAverage:
    """Shadow weights, with the decay ramped up so short phases are not wasted.

    A fixed 0.999 over a three-epoch phase leaves the average still mostly at
    its initialisation. The ramp reaches half weight by step 10 and the nominal
    decay only once there are enough steps for it to mean anything.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.updates = 0
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    def effective_decay(self):
        return min(self.decay, (1 + self.updates) / (10 + self.updates))

    @torch.no_grad()
    def update(self, model):
        decay = self.effective_decay()
        for name, parameter in model.named_parameters():
            if name in self.shadow and parameter.requires_grad:
                self.shadow[name].mul_(decay).add_(
                    parameter.detach(), alpha=1 - decay)
        self.updates += 1

    @torch.no_grad()
    def swap_in(self, model):
        backup = {}
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                backup[name] = parameter.detach().clone()
                parameter.data.copy_(self.shadow[name])
        return backup

    @torch.no_grad()
    def swap_out(self, model, backup):
        for name, parameter in model.named_parameters():
            if name in backup:
                parameter.data.copy_(backup[name])


def gradients_are_finite(model):
    """True when every gradient in the model is finite.

    It has to run after `backward` and before `clip_grad_norm_`: clipping a
    set of gradients whose total norm is NaN turns every one of them into NaN,
    so after the clip there is nothing left to detect.
    """
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            return False
    return True


def parameter_groups(model, config, phase):
    """Layer-wise learning rate decay over the unfrozen backbone blocks.

    Deeper blocks encode more general features and are moved less. In phase one
    nothing but the new modules moves at all. Depth is counted from the top of
    each backbone, whatever its number of blocks.
    """
    custom, backbone = [], {}
    depths = model.block_depths()

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if model.is_custom(name):
            custom.append(parameter)
            continue
        if id(parameter) not in depths:
            raise RuntimeError(f"{name} is trainable but sits in no block.")
        backbone.setdefault(depths[id(parameter)], []).append(parameter)

    if phase == 1:
        return [{
            "params": custom, "lr": config.phase1_lr_head,
            "weight_decay": config.weight_decay_head, "name": "head",
        }]

    groups = [{
        "params": params,
        "lr": config.phase2_lr_backbone * (config.llrd_decay ** depth),
        "weight_decay": config.weight_decay_backbone,
        "name": f"backbone_depth_{depth}",
    } for depth, params in sorted(backbone.items())]
    groups.append({
        "params": custom, "lr": config.phase2_lr_head,
        "weight_decay": config.weight_decay_head, "name": "head",
    })
    return groups


def _batch(batch, device):
    return (batch["input_ids"].to(device), batch["attention_mask"].to(device),
            batch["pixel_values"].to(device))


def train_phase(model, train_loader, val_loader, config, phase, pos_weight,
                device, history=None, verbose=True):
    """One phase. Returns the history, the best weights and their val AUROC.

    The best weights are the tuned parameters only, kept in memory: a full
    state dict per epoch would mostly be copies of frozen CLIP.
    """
    model.to(device)
    epochs = config.phase1_epochs if phase == 1 else config.phase2_epochs
    history = history or {
        "epoch": [], "phase": [], "loss": [], "val_auroc": [],
        "skipped_steps": [], "seconds": [],
    }

    optimiser = torch.optim.AdamW(
        parameter_groups(model, config, phase), betas=(0.9, 0.999), eps=1e-6)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = get_cosine_schedule_with_warmup(
        optimiser,
        num_warmup_steps=max(1, int(config.warmup_ratio * steps_per_epoch * epochs)),
        num_training_steps=steps_per_epoch * epochs,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    ema = ExponentialMovingAverage(model, config.ema_decay) if config.use_ema else None

    trainable, total = model.trainable_parameters()
    if verbose:
        print(f"\nPhase {phase}: {trainable:,} / {total:,} trainable "
              f"({100 * trainable / total:.2f}%)")

    best_auroc, best_state, no_improvement = -1.0, None, 0

    for epoch in range(epochs):
        model.train()
        started = time.time()
        running, counted, skipped = 0.0, 0, 0

        for batch in train_loader:
            labels = batch["label"].to(device)
            targets = (
                labels * (1 - config.label_smoothing)
                + 0.5 * config.label_smoothing
            )

            logits = model(*_batch(batch, device))
            loss = criterion(logits.float(), targets)

            optimiser.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()

            # Clipping a NaN norm poisons every gradient, and the optimiser
            # then writes that into the weights, so check first.
            if not gradients_are_finite(model):
                skipped += 1
                optimiser.zero_grad(set_to_none=True)
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimiser.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)

            running += loss.item()
            counted += 1

        if counted == 0:
            raise RuntimeError(
                f"Every step of epoch {epoch + 1} was skipped as non-finite. "
                "The model is not recoverable from here: lower the learning "
                "rate or unfreeze fewer layers."
            )

        backup = ema.swap_in(model) if ema is not None else None
        scores = evaluate(model, val_loader, device)
        val_auroc = auroc(scores["label"], scores["probability"])
        improved = not math.isnan(val_auroc) and val_auroc > best_auroc
        if improved:
            best_auroc, best_state, no_improvement = (
                val_auroc, model.tuned_state_dict(), 0)
        else:
            no_improvement += 1
        if ema is not None:
            ema.swap_out(model, backup)

        history["epoch"].append(epoch + 1)
        history["phase"].append(phase)
        history["loss"].append(running / counted)
        history["val_auroc"].append(val_auroc)
        history["skipped_steps"].append(skipped)
        history["seconds"].append(round(time.time() - started, 1))

        if verbose:
            note = f"  skipped {skipped}" if skipped else ""
            best = "  *" if improved else ""
            print(f"  epoch {epoch + 1:2d}/{epochs}  loss {running / counted:.4f}  "
                  f"val AUROC {val_auroc:.4f}  {history['seconds'][-1]:.0f}s"
                  f"{note}{best}")

        if phase == 2 and no_improvement >= config.patience:
            if verbose:
                print(f"    stopped early after {no_improvement} epochs "
                      "without improvement")
            break

    return history, best_state, best_auroc


def fit(model, loaders, config, pos_weight, device, verbose=True):
    """Both phases. Leaves the best weights of either phase in `model`."""
    model.freeze_backbones()
    history, best_state, phase1 = train_phase(
        model, loaders["train"], loaders["val"], config, 1, pos_weight,
        device, verbose=verbose)
    # Phase two starts from the best phase-one weights, not the last ones.
    model.load_tuned_state_dict(best_state)
    summary = {"phase1_val_auroc": phase1, "phase2_val_auroc": None,
               "best_phase": 1}

    if config.phase2_epochs > 0 and config.unfreeze_top > 0:
        model.unfreeze_top(config.unfreeze_top)
        history, state, phase2 = train_phase(
            model, loaders["train"], loaders["val"], config, 2, pos_weight,
            device, history=history, verbose=verbose)
        summary["phase2_val_auroc"] = phase2
        if state is not None and phase2 > phase1:
            best_state, summary["best_phase"] = state, 2

    model.load_tuned_state_dict(best_state)
    summary["skipped_steps"] = int(sum(history["skipped_steps"]))
    return history, summary
