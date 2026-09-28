"""A stand-in for the Hateful Memes directory, for the smoke run and the tests.

The real dataset cannot be redistributed and needs a Kaggle account, so a
fake one with the same layout lets the whole pipeline run on a fresh clone and
in CI: coloured squares for images, random words for text, alternating labels.
It exercises the code. It says nothing about the model.
"""
import random
from pathlib import Path

import pandas as pd
from PIL import Image

WORDS = (
    "cat dog tree river mountain city friend picnic morning rain window "
    "garden road music coffee paper ocean light family weekend"
).split()


def write_fake_dataset(root, n_train=64, n_dev=32, seed=0):
    """Write train/dev splits, with and without captions, and their images."""
    rng = random.Random(seed)
    root = Path(root)
    (root / "img").mkdir(parents=True, exist_ok=True)

    for split, count in (("train", n_train), ("dev", n_dev)):
        rows = []
        for index in range(count):
            ident = f"{split}{index:05d}"
            colour = tuple(rng.randrange(256) for _ in range(3))
            Image.new("RGB", (64, 64), colour).save(
                root / "img" / f"{ident}.png")
            rows.append({
                "id": ident,
                "img": f"img/{ident}.png",
                "label": index % 2,
                "text": " ".join(rng.choices(WORDS, k=rng.randint(2, 12))),
                "caption": f"a picture of a {rng.choice(WORDS)}",
            })
        frame = pd.DataFrame(rows)
        frame.to_json(root / f"{split}_captioned.jsonl",
                      orient="records", lines=True)
        frame.drop(columns="caption").to_json(
            root / f"{split}.jsonl", orient="records", lines=True)
    return root


def write_tiny_text_model(root, tokenizer_name):
    """A randomly initialised two-block XLM-RoBERTa, saved like a Hub model.

    It borrows the tokenizer of `tokenizer_name`, since only the vocabulary
    size has to agree. That is enough to exercise the non-CLIP text path,
    masked-mean pooling and `encoder.layer`, without downloading 1 GB.
    """
    from transformers import AutoTokenizer, XLMRobertaConfig, XLMRobertaModel

    root = Path(root)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    config = XLMRobertaConfig(
        vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=37,
        max_position_embeddings=tokenizer.model_max_length + 4,
        pad_token_id=tokenizer.pad_token_id,
    )
    XLMRobertaModel(config).save_pretrained(root)
    tokenizer.save_pretrained(root)
    return root
