"""Every hyperparameter in one dataclass, and the ablations as named overrides.

The reason beside a value is there when the value was chosen rather than
inherited.
"""
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # Backbones. Both are CLIP's, so the text and the image are embedded by
    # encoders that were trained to put a caption and its picture at the same
    # point. The clash below measures the distance between those two points,
    # which only means something if the two encoders share a space. The
    # dataset is English, so a multilingual text encoder buys nothing here;
    # XLM-RoBERTa is kept as the `xlmr` ablation to show what the shared space
    # is worth.
    text_model: str = "openai/clip-vit-base-patch32"
    vision_model: str = "openai/clip-vit-base-patch32"
    # CLIP ViT-B/32 projects both modalities to 512. Matching it lets the new
    # projections start from CLIP's own, so the two pooled vectors are aligned
    # from the first step rather than having to learn it.
    hidden_size: int = 512
    attention_heads: int = 8

    # What the fusion is made of. Each ablation switches one of these.
    #   gated         cross-modal attention, then the entropy-conditioned gate
    #   concat        the two pooled vectors side by side, nothing else
    #   concat_clash  the two pooled vectors and the clash, nothing else
    #   text          the text encoder alone
    #   image         the image encoder alone
    # `use_clash` and `use_entropy` only change the gated fusion.
    fusion: str = "gated"
    use_clash: bool = True
    use_entropy: bool = True

    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    checkpoint_dir: Path = field(default_factory=lambda: ROOT / "checkpoints")
    # Capped by the tokenizer's own limit, which is 77 for CLIP.
    max_text_length: int = 128
    use_captions: bool = True
    # Model selection and early stopping read this many memes held out from
    # train. The official dev split is only ever scored once, at the end.
    val_size: int = 500
    split_seed: int = 0

    batch_size: int = 32
    num_workers: int = 4

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

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        if self.fusion not in ("gated", "concat", "concat_clash", "text",
                               "image"):
            raise ValueError(f"Unknown fusion {self.fusion!r}.")

    @property
    def uses_text(self):
        return self.fusion != "image"

    @property
    def uses_image(self):
        return self.fusion != "text"

    def to_dict(self):
        return {key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(self).items()}


# One change each against `full`, so a difference in the results table is
# caused by the thing named in its row.
VARIANTS = {
    "full": {},
    "no_entropy": {"use_entropy": False},
    "no_clash": {"use_clash": False},
    "concat": {"fusion": "concat"},
    # The direct test of the premise: does an explicit gap add anything to a
    # classifier that already sees both vectors?
    "concat_clash": {"fusion": "concat_clash"},
    # Captions off, or a "text only" model would see the image through BLIP.
    "text_only": {"fusion": "text", "use_captions": False},
    "image_only": {"fusion": "image"},
    "no_captions": {"use_captions": False},
    "xlmr": {"text_model": "xlm-roberta-base"},
}


def config_for(variant, **overrides):
    if variant not in VARIANTS:
        raise ValueError(
            f"Unknown variant {variant!r}; choose from {', '.join(VARIANTS)}.")
    settings = {**VARIANTS[variant], **overrides}
    return Config(**settings)
