"""Generate an image caption for every meme, once, with BLIP.

    python scripts/captions.py

A meme's text says one thing and its picture shows another, and the encoder
only ever reads the text. Captioning the image gives the text encoder something
to disagree with: the input becomes "<overlaid text> [SEP] <what the picture
shows>", and the clash feature downstream has two descriptions of the same meme
to compare.

This writes `train_captioned.jsonl` and `dev_captioned.jsonl` beside the
originals and then gets out of the way. It is slow, it only has to run once,
and the training script picks the captioned files up automatically if they are
there.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import BlipForConditionalGeneration, BlipProcessor  # noqa: E402

MODEL = "Salesforce/blip-image-captioning-base"


def caption_file(source, image_root, destination, processor, model, device,
                 batch_size=32, limit=None):
    if destination.exists():
        print(f"  {destination.name} already exists, skipping")
        return

    frame = pd.read_json(source, lines=True)
    if limit:
        frame = frame.head(limit)

    captions, failures = [], 0
    for start in range(0, len(frame), batch_size):
        chunk = frame.iloc[start:start + batch_size]
        images, keep = [], []
        for position, path in enumerate(chunk["img"]):
            try:
                images.append(Image.open(image_root / str(path)).convert("RGB"))
                keep.append(position)
            except (OSError, ValueError):
                failures += 1

        produced = [""] * len(chunk)
        if images:
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=30)
            for position, sequence in zip(keep, output):
                produced[position] = processor.decode(
                    sequence, skip_special_tokens=True)
        captions.extend(produced)

        done = min(start + batch_size, len(frame))
        print(f"  {done}/{len(frame)}", end="\r", flush=True)

    frame["caption"] = captions
    frame.to_json(destination, orient="records", lines=True)
    print(f"\n  wrote {destination}"
          + (f", {failures} images unreadable" if failures else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.data)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Captioning with {MODEL} on {device}")

    processor = BlipProcessor.from_pretrained(MODEL)
    model = BlipForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    for split in ("train", "dev"):
        source = root / f"{split}.jsonl"
        if not source.exists():
            print(f"  {source} is missing, skipping")
            continue
        caption_file(
            source, root, root / f"{split}_captioned.jsonl",
            processor, model, device, args.batch_size, args.limit,
        )


if __name__ == "__main__":
    main()
