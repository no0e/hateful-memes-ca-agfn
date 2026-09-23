"""Two-phase training, and the guard that makes phase two survive.

The earlier version of this project trained phase one to an AUROC of 0.611 and
then died four lines into phase two, on ten consecutive non-finite losses. The
diagnosis is worth writing down, because the bug was in the safety net rather
than in the model.

That loop checked whether the **loss** was finite and skipped the step if it was
not. At the first step of phase two the loss was finite. The backward pass
through the newly unfrozen backbone produced non-finite **gradients**.
`clip_grad_norm_` computed a total norm of NaN, divided by it, and wrote NaN
into every gradient in the model. The optimiser then wrote those NaNs into the
weights. From the next step onward every forward pass returned NaN, the loss
check fired on all of them, and the run aborted ten steps later having already
lost the model.

The fix is one check in the right place: gradients are tested for finiteness
after the backward pass and before the clip. A bad batch is dropped and the
weights are never touched. `skipped_steps` counts how often that happens,
because a guard that silently eats half the batches is its own kind of failure.
"""
import copy
import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import get_cosine_schedule_with_warmup


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

    This is the check the previous version was missing. It has to run after
    `backward` and before `clip_grad_norm_`, because clipping a set of
    gradients whose norm is NaN turns all of them into NaN.
    """
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            return False
    return True


def parameter_groups(model, config, phase):
    """Layer-wise learning rate decay over the unfrozen backbone blocks.

    Deeper blocks encode more general features and are moved less. In phase one
    nothing but the new modules moves at all.
    """
    custom, backbone = [], {}

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(key in name for key in model.CUSTOM_MODULES):
            custom.append(parameter)
            continue
        for marker, tag in (("encoder.layer.", "text"),
                            ("encoder.layers.", "vision")):
            if marker in name:
                index = int(name.split(marker)[1].split(".")[0])
                backbone.setdefault(f"{tag}_{index}", {
                    "params": [], "index": index}) ["params"].append(parameter)
                break

    if phase == 1:
        return [{
            "params": custom, "lr": config.phase1_lr_head,
            "weight_decay": config.weight_decay_head, "name": "head",
        }]

    groups = []
    for name, group in backbone.items():
        offset = 11 - group["index"]
        groups.append({
            "params": group["params"],
            "lr": config.phase2_lr_backbone * (config.llrd_decay ** offset),
            "weight_decay": config.weight_decay_backbone,
            "name": name,
        })
    groups.append({
        "params": custom, "lr": config.phase2_lr_head,
        "weight_decay": config.weight_decay_head, "name": "head",
    })
    return groups


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    probabilities, labels = [], []
    for batch in loader:
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["pixel_values"].to(device),
        )
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(batch["label"].numpy())

    probabilities = np.concatenate(probabilities)
    labels = np.concatenate(labels)
    predictions = (probabilities > threshold).astype(int)
    try:
        auroc = float(roc_auc_score(labels, probabilities))
    except ValueError:  # a batch of one class only
        auroc = float("nan")
    return {
        "auroc": auroc,
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro")),
    }


def train_phase(model, train_loader, val_loader, config, phase, pos_weight,
                device, checkpoint_dir, history=None, verbose=True):
    """One phase. Returns the history and the path of the best checkpoint."""
    model.to(device)
    epochs = config.phase1_epochs if phase == 1 else config.phase2_epochs
    history = history or {
        "epoch": [], "phase": [], "loss": [], "auroc": [], "accuracy": [],
        "f1_macro": [], "skipped_steps": [],
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

    best_auroc, best_path, no_improvement = -1.0, None, 0
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running, counted, skipped = 0.0, 0, 0

        for batch in train_loader:
            labels = batch["label"].to(device)
            targets = (
                labels * (1 - config.label_smoothing)
                + 0.5 * config.label_smoothing
            )

            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["pixel_values"].to(device),
            )
            loss = criterion(logits.float(), targets)

            optimiser.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()

            # The check that was missing. Clipping a NaN norm poisons every
            # gradient, and the optimiser then writes that into the weights.
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

        average_loss = running / counted
        if ema is not None:
            backup = ema.swap_in(model)
            metrics = evaluate(model, val_loader, device)
            ema.swap_out(model, backup)
        else:
            metrics = evaluate(model, val_loader, device)

        history["epoch"].append(epoch + 1)
        history["phase"].append(phase)
        history["loss"].append(average_loss)
        history["skipped_steps"].append(skipped)
        for key in ("auroc", "accuracy", "f1_macro"):
            history[key].append(metrics[key])

        if verbose:
            note = f"  skipped {skipped}" if skipped else ""
            print(
                f"  epoch {epoch + 1:2d}/{epochs}  loss {average_loss:.4f}  "
                f"AUROC {metrics['auroc']:.4f}  acc {metrics['accuracy']:.4f}  "
                f"F1 {metrics['f1_macro']:.4f}{note}"
            )

        if not math.isnan(metrics["auroc"]) and metrics["auroc"] > best_auroc:
            best_auroc, no_improvement = metrics["auroc"], 0
            if ema is not None:
                backup = ema.swap_in(model)
                state = copy.deepcopy(model.state_dict())
                ema.swap_out(model, backup)
            else:
                state = copy.deepcopy(model.state_dict())

            best_path = checkpoint_dir / f"phase{phase}_best.pt"
            torch.save({
                "state_dict": state, "epoch": epoch, "phase": phase,
                "auroc": best_auroc, "history": history,
                "config": vars(config),
            }, best_path)
            if verbose:
                print(f"    new best AUROC {best_auroc:.4f}")
        else:
            no_improvement += 1
            if phase == 2 and no_improvement >= config.patience:
                if verbose:
                    print(f"    stopped early after {no_improvement} epochs "
                          "without improvement")
                break

    return history, best_path, best_auroc
