"""Experiment tracking with Weights & Biases, off unless asked for.

    python scripts/train.py --wandb                      # online, project ca-agfn
    WANDB_MODE=offline python scripts/train.py --wandb   # logged locally, synced later

Every run logs its full config, one point per epoch (loss, validation AUROC,
skipped steps, seconds) with the phase as a field, and the test scores as the
run summary, so the 27 ablation runs can be compared side by side in one
project. Without --wandb nothing is imported and nothing leaves the machine:
the JSON under results/ stays the record either way.
"""


class Tracker:
    """A no-op by default; a W&B run when `enabled`."""

    def __init__(self, enabled=False, project="ca-agfn", name=None, group=None,
                 config=None, tags=None):
        self.run = None
        if not enabled:
            return
        try:
            import wandb
        except ImportError as error:
            raise SystemExit(
                "--wandb needs the wandb package: pip install -e '.[track]'"
            ) from error
        self.run = wandb.init(project=project, name=name, group=group,
                              config=config or {}, tags=tags or [],
                              reinit="finish_previous")
        # Epochs restart at 1 in phase two; a global step keeps them in order.
        self.run.define_metric("global_epoch")
        self.run.define_metric("*", step_metric="global_epoch")
        self.global_epoch = 0

    @property
    def enabled(self):
        return self.run is not None

    def epoch(self, phase, epoch, **metrics):
        if not self.enabled:
            return
        self.global_epoch += 1
        self.run.log({"global_epoch": self.global_epoch, "phase": phase,
                      "epoch": epoch, **metrics})

    def summary(self, values, prefix=""):
        if not self.enabled:
            return
        for key, value in values.items():
            if isinstance(value, (int, float, bool)) or value is None:
                self.run.summary[f"{prefix}{key}"] = value

    def finish(self):
        if self.enabled:
            self.run.finish()
