"""Run the trained model on a constructed meme, and draw the result.

    python scripts/example.py --data ~/work/memes-data

The same line of text over two different images. On the calm mountain it reads
literally; over the same mountain erupting it reads as its opposite. That is
the structure the whole benchmark is built on, and the structure this
architecture exists to catch: neither the text alone nor the image alone
distinguishes the two, only the relationship between them does.

Both photographs are public domain, from the United States Geological Survey,
and their provenance is in docs/example/PROVENANCE.md. They are not from the
Hateful Memes benchmark, whose licence forbids redistributing its images and
whose contents are not something to put on a public page. So this example is
constructed. The numbers on it are not: they are what the trained model
returns for these two inputs.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from transformers import AutoTokenizer, CLIPImageProcessor  # noqa: E402

from ca_agfn.config import Config  # noqa: E402
from ca_agfn.model import CAAGFN  # noqa: E402

CAPTION = "everything is under control"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#898781"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"


def _font(size):
    for candidate in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial_Bold.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def meme(path, text, width=520, height=520):
    """Compose the classic form: the image, the line of text across the bottom.

    Both panels come out the same size, cropped to a square around the centre,
    so the two sit level beside each other however the originals were framed.
    """
    image = Image.open(path).convert("RGB")

    scale = max(width / image.width, height / image.height)
    image = image.resize(
        (max(width, int(image.width * scale)),
         max(height, int(image.height * scale))), Image.LANCZOS)
    left = (image.width - width) // 2
    top = (image.height - height) // 2
    image = image.crop((left, top, left + width, top + height))

    draw = ImageDraw.Draw(image)
    text = text.upper()

    # Shrink until it fits with a margin, rather than letting it run off the
    # edge, which is what a fixed size does as soon as the caption gets longer.
    size = width // 12
    while size > 10:
        font = _font(size)
        box = draw.textbbox((0, 0), text, font=font)
        if box[2] - box[0] <= width * 0.92:
            break
        size -= 1

    box = draw.textbbox((0, 0), text, font=font)
    x = (image.width - (box[2] - box[0])) / 2 - box[0]
    y = image.height - (box[3] - box[1]) - size * 0.9

    # White on a black outline, which is how meme text stays readable over
    # anything. Drawn by hand because stroke_width is not in every Pillow.
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            if dx or dy:
                draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    return image


@torch.no_grad()
def score(model, tokenizer, processor, image, text, device, max_length=128):
    encoded = tokenizer([text], padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt")
    pixels = processor(images=image, return_tensors="pt")

    logits, extra = model(
        encoded["input_ids"].to(device),
        encoded["attention_mask"].to(device),
        pixels["pixel_values"].to(device),
        return_gate=True,
    )
    return {
        "probability": float(torch.sigmoid(logits.float())[0]),
        "gate": float(extra["gate"][0]),
        "entropy": float(extra["entropy"][0]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--text", default=CAPTION)
    parser.add_argument("--scores", default=None,
                        help="Render from a scores file instead of loading the "
                             "model. The file is written by a normal run; this "
                             "redraws the figure on a machine that has the "
                             "photographs but not the 1.4 GB checkpoint.")
    parser.add_argument("--out", default=str(ROOT / "docs" / "example.jpg"))
    args = parser.parse_args()

    config = Config()
    if args.data:
        config.data_dir = Path(args.data)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    cached = None
    if args.scores:
        cached = json.loads(Path(args.scores).read_text(encoding="utf-8"))
        print(f"Rendering from {args.scores}, measured on "
              f"{cached.get('device', 'an unrecorded device')}.\n")
    else:
        checkpoint = Path(
            args.checkpoint or (config.checkpoint_dir / "phase2_best.pt"))
        if not checkpoint.exists():
            raise SystemExit(
                f"No checkpoint at {checkpoint}. Run scripts/train.py first, "
                "or pass --scores to redraw from a previous run."
            )

        tokenizer = AutoTokenizer.from_pretrained(config.text_model)
        processor = CLIPImageProcessor.from_pretrained(config.vision_model)
        model = CAAGFN(config)
        model.load_state_dict(
            torch.load(checkpoint, map_location=device,
                       weights_only=False)["state_dict"])
        model.to(device).eval()

    source = ROOT / "docs" / "example"
    panels = [
        ("The mountain, before", source / "calm.jpg"),
        ("The same mountain, erupting", source / "erupting.jpg"),
    ]

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    })
    figure, axes = plt.subplots(1, 2, figsize=(11, 6.8))

    results = []
    for ax, (title, path) in zip(axes, panels):
        composed = meme(path, args.text)
        measured = (cached["panels"][title] if cached else
                    score(model, tokenizer, processor, composed, args.text,
                          device))
        results.append((title, measured))

        ax.imshow(composed)
        ax.axis("off")
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=INK)
        ax.text(
            0.5, -0.035,
            f"entropy {measured['entropy']:.3f}      "
            f"gate {measured['gate']:.3f}      "
            f"P(hateful) {measured['probability']:.3f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=10,
            color=SERIES_1, fontweight="bold",
        )

    figure.suptitle(
        f'Same words, two images: "{args.text}"',
        fontsize=13, fontweight="bold", color=INK, x=0.012, ha="left", y=0.98)
    figure.text(
        0.012, 0.925,
        "A constructed example: the photographs are public domain (USGS) and "
        "no benchmark image is reproduced.\nThe three figures under each "
        "panel are what the trained model returns for that input.",
        fontsize=8.5, color=INK_MUTED, ha="left", va="top")
    figure.tight_layout(rect=[0, 0.075, 1, 0.87])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=150)
    plt.close(figure)

    print(f"Wrote {out}\n")
    for title, measured in results:
        print(f"  {title:<32} entropy {measured['entropy']:.3f}  "
              f"gate {measured['gate']:.3f}  "
              f"P(hateful) {measured['probability']:.3f}")

    difference = abs(results[0][1]["gate"] - results[1][1]["gate"])
    print(
        f"\n  gate difference between the two images: {difference:.3f}\n"
        "  The text is byte for byte identical, so any difference comes from "
        "the image alone."
    )


if __name__ == "__main__":
    main()
