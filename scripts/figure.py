"""Draw the results figure from a trained checkpoint.

    python scripts/figure.py --data ~/work/memes-data

Four panels, and the first one is the point: the architecture claims that the
entropy of the text's attention decides how much the text is trusted, so that
claim is plotted against the gate the model actually produced on every
validation meme. An architecture diagram would show what was intended. This
shows what happened.

No dataset image appears in the output. The Hateful Memes images are under
Meta's research licence and derived from stock photography, so redistributing
them in a public repository is not ours to do, and the benchmark is built
around hateful content besides. Where a meme is shown, it is shown as its
extracted text and its generated caption, which is exactly what the model
reads anyway.
"""
import argparse
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from transformers import AutoTokenizer, CLIPImageProcessor  # noqa: E402

from ca_agfn.config import Config  # noqa: E402
from ca_agfn.data import HatefulMemes, split_file  # noqa: E402
from ca_agfn.model import CAAGFN  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_1 = "#2a78d6"   # blue
SERIES_2 = "#eb6834"   # orange


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "sans-serif"], "font.size": 9,
        "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "axes.labelsize": 9,
        "axes.labelcolor": INK_SECONDARY, "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.7, "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "legend.frameon": False, "legend.fontsize": 8.5,
    })


def strip(ax, keep=("left", "bottom")):
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


@torch.no_grad()
def collect(model, dataset, device, batch_size=32):
    """Probability, gate and entropy for every meme in the split."""
    model.eval()
    probabilities, gates, entropies = [], [], []

    for start in range(0, len(dataset), batch_size):
        rows = [dataset[i] for i in range(start, min(start + batch_size,
                                                     len(dataset)))]
        batch = {
            key: torch.stack([row[key] for row in rows]).to(device)
            for key in ("input_ids", "attention_mask", "pixel_values")
        }
        logits, extra = model(**batch, return_gate=True)
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
        gates.append(extra["gate"].float().cpu().numpy())
        entropies.append(extra["entropy"].float().cpu().numpy())

    return (np.concatenate(probabilities), np.concatenate(gates),
            np.concatenate(entropies))


def panel_gate_against_entropy(ax, entropy, gate):
    """The architecture's claim, measured rather than asserted."""
    ax.scatter(entropy, gate, s=14, color=SERIES_1, alpha=0.35,
               edgecolor="none", zorder=2)

    # A trend line, because 500 translucent dots can hide a weak relationship.
    order = np.argsort(entropy)
    window = max(15, len(entropy) // 20)
    smoothed = np.convolve(gate[order], np.ones(window) / window, mode="valid")
    ax.plot(entropy[order][window - 1:], smoothed, color=SERIES_2, linewidth=2,
            zorder=4, label="Rolling mean")

    correlation = float(np.corrcoef(entropy, gate)[0, 1])
    ax.annotate(f"r = {correlation:+.2f}", (0.03, 0.05),
                xycoords="axes fraction", fontsize=10, color=INK,
                fontweight="bold")

    ax.set_xlabel("Entropy of the text's attention over the image")
    ax.set_ylabel("Gate: weight given to the text")
    ax.set_title("Does the entropy gate do what it claims?")
    ax.legend(loc="upper right")
    strip(ax)


def panel_roc(ax, labels, probabilities):
    false_positive, true_positive, _ = roc_curve(labels, probabilities)
    auroc = roc_auc_score(labels, probabilities)

    ax.plot([0, 1], [0, 1], color=BASELINE, linewidth=1.2,
            linestyle=(0, (3, 3)), zorder=2, label="Chance")
    ax.plot(false_positive, true_positive, color=SERIES_1, linewidth=2,
            zorder=3, label=f"CA-AGFN, AUROC {auroc:.3f}")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Validation ROC, 500 memes")
    ax.legend(loc="lower right")
    strip(ax)


def panel_gate_by_label(ax, gate, labels):
    """Whether the gate settles differently on hateful and benign memes."""
    groups = [("Not hateful", gate[labels == 0], SERIES_1),
              ("Hateful", gate[labels == 1], SERIES_2)]

    for row, (name, values, colour) in enumerate(groups):
        low, mid, high = np.percentile(values, [25, 50, 75])
        ax.hlines(row, low, high, color=colour, linewidth=7, alpha=0.30,
                  zorder=2)
        ax.scatter([mid], [row], s=80, color=colour, zorder=4,
                   edgecolor=SURFACE, linewidth=2, label=name)
        ax.annotate(f"{mid:.2f}", (mid, row), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=9, color=INK,
                    fontweight="bold")
        ax.annotate(f"n={len(values)}", (high, row),
                    textcoords="offset points", xytext=(10, -3), fontsize=7.5,
                    color=INK_MUTED)

    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([name for name, _, _ in groups])
    ax.set_ylim(-0.7, len(groups) - 0.3)
    ax.set_xlabel("Gate: weight given to the text")
    ax.set_title("Where the gate settles, by label")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", ncol=2)
    strip(ax)


def panel_examples(ax, frame, probabilities, gates, entropies, n=3):
    """Real validation memes, as the model reads them.

    Benign memes only, and the picture is represented by the caption the model
    was given. Reproducing the images would redistribute a licensed research
    dataset built around hateful content.
    """
    ax.axis("off")
    ax.set_title("Three real memes, as the model reads them", loc="left")

    benign = np.flatnonzero(frame["label"].to_numpy() == 0)
    # Spread across the entropy range so the examples are not all alike.
    chosen = benign[np.argsort(entropies[benign])][
        np.linspace(0, len(benign) - 1, n).astype(int)]

    y = 0.95
    for index in chosen:
        row = frame.iloc[index]
        text = str(row.get("text", ""))[:90]
        caption = str(row.get("caption", ""))[:70]

        ax.text(0.0, y, "text", fontsize=7.5, color=INK_MUTED,
                transform=ax.transAxes)
        ax.text(0.11, y, textwrap.shorten(text, 88, placeholder=" ..."),
                fontsize=8.5, color=INK, transform=ax.transAxes)
        y -= 0.075
        ax.text(0.0, y, "image", fontsize=7.5, color=INK_MUTED,
                transform=ax.transAxes)
        ax.text(0.11, y, textwrap.shorten(caption or "(no caption)", 88,
                                          placeholder=" ..."),
                fontsize=8.5, color=INK_SECONDARY, style="italic",
                transform=ax.transAxes)
        y -= 0.075
        ax.text(0.11, y,
                f"entropy {entropies[index]:.2f}    "
                f"gate {gates[index]:.2f}    "
                f"P(hateful) {probabilities[index]:.2f}    "
                f"truth: not hateful",
                fontsize=8.5, color=SERIES_1, fontweight="bold",
                transform=ax.transAxes)
        y -= 0.13


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=str(ROOT / "docs" / "results.png"))
    args = parser.parse_args()

    config = Config()
    if args.data:
        config.data_dir = Path(args.data)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    checkpoint_path = Path(
        args.checkpoint or (config.checkpoint_dir / "phase2_best.pt"))
    if not checkpoint_path.exists():
        raise SystemExit(
            f"No checkpoint at {checkpoint_path}. Run scripts/train.py first."
        )

    tokenizer = AutoTokenizer.from_pretrained(config.text_model)
    processor = CLIPImageProcessor.from_pretrained(config.vision_model)
    dataset = HatefulMemes(
        split_file(config.data_dir, "dev", config.use_captions),
        config.data_dir, tokenizer, processor, config.max_text_length,
        train=False, use_captions=config.use_captions,
    )

    model = CAAGFN(config)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device,
                   weights_only=False)["state_dict"])
    model.to(device)

    probabilities, gates, entropies = collect(model, dataset, device)
    labels = dataset.labels.numpy()

    style()
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
    panel_gate_against_entropy(axes[0, 0], entropies, gates)
    panel_roc(axes[0, 1], labels, probabilities)
    panel_gate_by_label(axes[1, 0], gates, labels)
    panel_examples(axes[1, 1], dataset.frame, probabilities, gates, entropies)

    figure.suptitle(
        "CA-AGFN on the Hateful Memes validation split",
        fontsize=13, fontweight="bold", color=INK, x=0.012, ha="left", y=0.985)
    figure.text(
        0.012, 0.945,
        "No dataset image is reproduced: the benchmark is licensed for "
        "research and built around hateful content. A meme is shown as the "
        "text and caption the model reads.",
        fontsize=8.5, color=INK_MUTED, ha="left")
    figure.tight_layout(rect=[0, 0, 1, 0.925])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=150)
    print(f"Wrote {out}")
    print(f"AUROC {roc_auc_score(labels, probabilities):.4f}   "
          f"gate against entropy r = {np.corrcoef(entropies, gates)[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
