"""Every hyperparameter in one dataclass, with the reason beside the ones that
were chosen rather than inherited.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # Backbones. XLM-RoBERTa rather than DeBERTa-v3: memes carry multilingual
    # and transliterated text, and DeBERTa-v3's disentangled attention was the
    # component that made the earlier phase-two runs fragile.
    text_model: str = "xlm-roberta-base"
    vision_model: str = "openai/clip-vit-base-patch32"
    hidden_size: int = 768

    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    checkpoint_dir: Path = field(default_factory=lambda: ROOT / "checkpoints")
    max_text_length: int = 128
    use_captions: bool = True

    batch_size: int = 32
    num_workers: int = 2

    # Phase 1 warms up the new modules against frozen backbones. Phase 2 opens
    # the top two blocks of each and moves them slowly.
    phase1_epochs: int = 3
    phase1_lr_head: float = 1e-4
    phase2_epochs: int = 10
    phase2_lr_head: float = 2e-5
    phase2_lr_backbone: float = 1e-6
    unfreeze_top: int = 2
    llrd_decay: float = 0.95
    patience: int = 3

    dropout: float = 0.4
    weight_decay_head: float = 0.01
    weight_decay_backbone: float = 0.05
    label_smoothing: float = 0.05
    grad_clip: float = 0.5
    warmup_ratio: float = 0.1

    use_ema: bool = True
    ema_decay: float = 0.999

    seed: int = 42
    # Off by default. The gradient guard in training.py makes a diverging run
    # survivable rather than fatal, but mixed precision on this architecture
    # was the thing that made it diverge in the first place, so turning it on
    # is a deliberate choice and not a default.
    mixed_precision: bool = False

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
