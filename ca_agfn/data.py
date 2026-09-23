"""The Hateful Memes dataset, tokenised once and read many times.

The dataset is Meta's Hateful Memes benchmark: 10,000 memes, each an image with
its overlaid text already extracted into a `text` field, labelled hateful or
not. It is deliberately built so that unimodal models fail: for most hateful
memes there exists a benign one with the same text over a different picture, or
the same picture under different text. That construction is the reason a fusion
model is worth building at all.

Text is tokenised in `__init__` rather than in `__getitem__`. There are only ten
thousand rows, so the whole tokenised corpus is a few megabytes, and doing it
once takes the tokeniser out of the training loop entirely.
"""
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_TRANSFORM = None


def train_transform():
    """Light augmentation, built on first use.

    Imported lazily so this module can be loaded for its path logic without
    pulling torchvision in. Light on purpose: the text inside a meme is part of
    the signal, and aggressive cropping or rotation destroys the thing the
    model has to read.
    """
    global _TRANSFORM
    if _TRANSFORM is None:
        from torchvision import transforms

        _TRANSFORM = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomRotation(degrees=5),
        ])
    return _TRANSFORM


class HatefulMemes(Dataset):
    """One meme: its tokenised text, its pixels, its label."""

    def __init__(self, jsonl, image_root, tokenizer, image_processor,
                 max_length=128, train=False, use_captions=True, limit=None):
        jsonl = Path(jsonl)
        if not jsonl.exists():
            raise FileNotFoundError(
                f"{jsonl} is missing. See the README for how to get the "
                "dataset; it needs a Kaggle or Meta account and cannot be "
                "redistributed here."
            )

        self.frame = pd.read_json(jsonl, lines=True)
        if limit:
            self.frame = self.frame.head(limit)
        self.image_root = Path(image_root)
        self.image_processor = image_processor
        self.train = train

        separator = tokenizer.sep_token or "[SEP]"
        has_captions = use_captions and "caption" in self.frame.columns

        def combine(row):
            text = str(row.get("text", "") or "")
            if has_captions:
                caption = str(row.get("caption", "") or "")
                if caption:
                    # The caption describes what the image shows; the text is
                    # what it says. Giving the encoder both is what lets it
                    # notice when they disagree.
                    return f"{text} {separator} {caption}"
            return text

        encoded = tokenizer(
            [combine(row) for _, row in self.frame.iterrows()],
            padding="max_length", truncation=True, max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]
        self.labels = torch.tensor(
            self.frame["label"].to_numpy(), dtype=torch.float32)
        self.paths = [self.image_root / str(p) for p in self.frame["img"]]
        self.has_captions = has_captions

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        try:
            image = Image.open(self.paths[index]).convert("RGB")
        except (OSError, ValueError):
            # A missing or truncated file becomes a blank image rather than
            # killing the epoch. The count is reported by `missing_images`, so
            # this never passes silently.
            image = Image.new("RGB", (224, 224))

        if self.train:
            image = train_transform()(image)

        pixels = self.image_processor(images=image, return_tensors="pt")
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "pixel_values": pixels["pixel_values"].squeeze(0),
            "label": self.labels[index],
        }

    def missing_images(self):
        """How many rows point at a file that is not there.

        Worth calling before a run: a dataset where a tenth of the images are
        blank trains perfectly well and scores badly for a reason nobody looks
        for.
        """
        return sum(1 for path in self.paths if not path.exists())

    def positive_weight(self):
        """Ratio of negatives to positives, for the loss.

        Hateful Memes is about 36% positive, so this is around 1.8. Without it
        the model learns that guessing "not hateful" is usually right.
        """
        positives = float(self.labels.sum())
        return torch.tensor(
            [(len(self.labels) - positives) / max(positives, 1.0)],
            dtype=torch.float32,
        )


def split_file(root, split, use_captions=True):
    """The captioned split if `scripts/captions.py` has run, else the plain one.

    Without this the captioning script writes `<split>_captioned.jsonl` and
    nothing ever reads it: the loader would go on opening `<split>.jsonl`, the
    caption column would be absent, and an expensive BLIP pass would silently
    make no difference to the model.
    """
    captioned = Path(root) / f"{split}_captioned.jsonl"
    if use_captions and captioned.exists():
        return captioned
    return Path(root) / f"{split}.jsonl"


def build_loaders(config, tokenizer, image_processor, limit=None):
    """Train and validation loaders, plus the positive weight."""
    root = Path(config.data_dir)
    train = HatefulMemes(
        split_file(root, "train", config.use_captions), root, tokenizer,
        image_processor, config.max_text_length, train=True,
        use_captions=config.use_captions, limit=limit,
    )
    validation = HatefulMemes(
        split_file(root, "dev", config.use_captions), root, tokenizer,
        image_processor, config.max_text_length, train=False,
        use_captions=config.use_captions, limit=limit,
    )

    missing = train.missing_images() + validation.missing_images()
    if missing:
        print(f"  warning: {missing} image files are missing and will be "
              "read as blank")

    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": True,
        "persistent_workers": config.num_workers > 0,
    }
    return (
        DataLoader(train, shuffle=True, drop_last=True, **common),
        DataLoader(validation, shuffle=False, **common),
        train.positive_weight(),
    )


def write_history(history, path):
    Path(path).write_text(json.dumps(history, indent=2), encoding="utf-8")
