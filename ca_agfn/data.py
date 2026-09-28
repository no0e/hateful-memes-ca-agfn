"""The Hateful Memes dataset, tokenised once and read many times.

The dataset is Meta's Hateful Memes benchmark: 10,000 memes, each an image with
its overlaid text already extracted into a `text` field, labelled hateful or
not. It is deliberately built so that unimodal models fail: for most hateful
memes there exists a benign one with the same text over a different picture, or
the same picture under different text. That construction is the reason a fusion
model is worth building at all.

Three splits come out of here, and keeping them apart is the point:

    train   the official train split, minus a stratified hold-out
    val     that hold-out, 500 memes: picks the best epoch and the threshold
    test    the official dev split, 500 memes: scored once, at the end

Selecting checkpoints on the dev split and then reporting the dev score would
flatter the number by however much the selection overfit.

Text is tokenised in `__init__` rather than in `__getitem__`. There are only ten
thousand rows, so the whole tokenised corpus is a few megabytes, and doing it
once takes the tokeniser out of the training loop entirely.
"""
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def augment(image):
    """Light colour jitter and a small rotation, and never a flip.

    The text inside a meme is part of the signal, and CLIP reads rendered text
    surprisingly well. A horizontal flip would mirror it and destroy what the
    augmentation is meant to preserve. `random` is reseeded per worker by the
    DataLoader.
    """
    for enhancer, spread in ((ImageEnhance.Brightness, 0.2),
                             (ImageEnhance.Contrast, 0.2),
                             (ImageEnhance.Color, 0.1)):
        image = enhancer(image).enhance(random.uniform(1 - spread, 1 + spread))
    return image.rotate(random.uniform(-5, 5),
                        resample=Image.Resampling.BILINEAR)


class HatefulMemes(Dataset):
    """One meme: its tokenised text, its pixels, its label."""

    def __init__(self, frame, image_root, tokenizer, image_processor,
                 max_length=128, train=False, use_captions=True,
                 load_images=True):
        self.frame = frame.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.image_processor = image_processor
        self.train = train
        self.load_images = load_images

        # CLIP's text encoder has 77 positions and nothing past them.
        max_length = min(max_length, tokenizer.model_max_length)
        # XLM-R has a separator token; CLIP does not, and "[SEP]" would be
        # read as three ordinary tokens.
        separator = tokenizer.sep_token or "."
        self.has_captions = use_captions and "caption" in self.frame.columns

        def combine(row):
            text = str(row.get("text", "") or "")
            if self.has_captions:
                caption = str(row.get("caption", "") or "")
                if caption:
                    # The caption describes what the image shows; the text is
                    # what it says.
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
        self.ids = [str(i) for i in self.frame.get(
            "id", pd.Series(range(len(self.frame))))]

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        if self.load_images:
            try:
                image = Image.open(self.paths[index]).convert("RGB")
            except (OSError, ValueError):
                # A missing or truncated file becomes a blank image rather than
                # killing the epoch. `missing_images` reports the count, so
                # this never passes silently.
                image = Image.new("RGB", (224, 224))
            if self.train:
                image = augment(image)
            pixels = self.image_processor(
                images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        else:
            # A text-only model never reads the image, so never decode it.
            pixels = torch.zeros(0)

        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "pixel_values": pixels,
            "label": self.labels[index],
        }

    def missing_images(self):
        """How many rows point at a file that is not there.

        A dataset where a tenth of the images are blank trains perfectly well
        and scores badly for a reason nobody looks for.
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
    nothing reads it, and an expensive BLIP pass silently changes nothing.
    """
    captioned = Path(root) / f"{split}_captioned.jsonl"
    if use_captions and captioned.exists():
        return captioned
    return Path(root) / f"{split}.jsonl"


def captions_available(root):
    """Captions for both splits, or for neither.

    Training on captioned text and testing on bare text would be a
    distribution shift introduced by the file system.
    """
    return all((Path(root) / f"{split}_captioned.jsonl").exists()
               for split in ("train", "dev"))


def read_split(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run scripts/fetch_data.py, or pass --smoke to "
            "run on generated data; the real dataset needs a Kaggle or Meta "
            "account and cannot be redistributed here."
        )
    return pd.read_json(path, lines=True)


def hold_out(frame, size, seed=0):
    """A stratified hold-out, identical for every run and every variant.

    `seed` is the split seed, not the training seed, so the seeds of an
    ablation vary the training and never the data it is scored on.
    """
    train, held = train_test_split(
        frame, test_size=size, stratify=frame["label"], random_state=seed)
    return train.reset_index(drop=True), held.reset_index(drop=True)


def load_frames(config):
    """Train, val and test frames, and whether captions are in them."""
    root = Path(config.data_dir)
    use_captions = config.use_captions
    if use_captions and not captions_available(root):
        print("  captions requested but not generated for both splits; "
              "run scripts/captions.py. Continuing without them.")
        use_captions = False

    train = read_split(split_file(root, "train", use_captions))
    test = read_split(split_file(root, "dev", use_captions))
    train, val = hold_out(train, config.val_size, config.split_seed)
    return train, val, test, use_captions


def build_loaders(config, tokenizer, image_processor, pin_memory=False):
    """Train, val and test loaders, the positive weight, and the caption flag."""
    train_frame, val_frame, test_frame, use_captions = load_frames(config)
    shared = {
        "image_root": config.data_dir, "tokenizer": tokenizer,
        "image_processor": image_processor,
        "max_length": config.max_text_length, "use_captions": use_captions,
        "load_images": config.uses_image,
    }
    datasets = {
        "train": HatefulMemes(train_frame, train=True, **shared),
        "val": HatefulMemes(val_frame, **shared),
        "test": HatefulMemes(test_frame, **shared),
    }

    if config.uses_image:
        missing = sum(d.missing_images() for d in datasets.values())
        if missing:
            print(f"  warning: {missing} image files are missing and will be "
                  "read as blank")

    # Workers exist to decode images, so a text-only model loads in-process: a
    # third of a second per epoch. Workers are spawned, not forked. By the
    # time a loader starts, this process holds CUDA, tokenizer and progress-bar
    # threads, and forking it deadlocked two of the first four runs on a T4,
    # at random, before their first epoch finished. The timeout turns any
    # other hang into an error instead of a stalled queue of runs.
    workers = config.num_workers if config.uses_image else 0
    common = {
        "batch_size": config.batch_size,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": workers > 0,
        "timeout": 600 if workers else 0,
        "multiprocessing_context": "spawn" if workers else None,
    }
    loaders = {
        name: DataLoader(dataset, shuffle=name == "train",
                         drop_last=name == "train", **common)
        for name, dataset in datasets.items()
    }
    return loaders, datasets["train"].positive_weight(), use_captions
