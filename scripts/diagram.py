"""Draw the architecture diagram, docs/pipeline.svg.

    python scripts/diagram.py

Drawn with matplotlib in the palette the result figures use, so it is
reproducible and can be kept in step with the model.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from ca_agfn.config import ROOT  # noqa: E402

SURFACE = "#fcfcfb"
GROUP = "#f2f1ec"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
EDGE = "#c3c2b7"
ARROW = "#898781"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

WIDTH, HEIGHT = 1250, 315


def box(ax, x, y, w, h, title=None, detail=None, edge=EDGE, face="white",
        weight=1.2, radius=8):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=weight, edgecolor=edge, facecolor=face, zorder=2))
    if title:
        middle = y + h / 2 - (7 if detail else 0)
        ax.text(x + w / 2, middle, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=INK, zorder=3)
    if detail:
        ax.text(x + w / 2, y + h / 2 + 10, detail, ha="center", va="center",
                fontsize=7.5, color=INK_MUTED, zorder=3)


def group(ax, x, y, w, h, title, detail):
    box(ax, x, y, w, h, face=GROUP, weight=1, radius=10)
    ax.text(x + 16, y + 20, title, fontsize=8.5, fontweight="bold",
            color=INK_SECONDARY, va="center")
    ax.text(x + 16, y + 38, detail, fontsize=7.5, style="italic",
            color=INK_MUTED, va="center")


def arrow(ax, start, end, label=None, dashed=False, offset=(0, -7),
          align="center"):
    ax.annotate("", xy=end, xytext=start, zorder=4, arrowprops={
        "arrowstyle": "-|>", "color": ARROW, "linewidth": 1.2,
        "linestyle": (0, (4, 3)) if dashed else "-", "mutation_scale": 11,
        "shrinkA": 0, "shrinkB": 0})
    if label:
        x = (start[0] + end[0]) / 2 + offset[0]
        y = (start[1] + end[1]) / 2 + offset[1]
        ax.text(x, y, label, fontsize=7.5, style="italic", color=INK_MUTED,
                ha=align, va="center", zorder=5,
                bbox={"boxstyle": "square,pad=0.15", "fc": GROUP, "ec": "none"})


def draw(out):
    figure = plt.figure(figsize=(WIDTH / 90, HEIGHT / 90))
    figure.patch.set_facecolor(SURFACE)
    ax = figure.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.axis("off")

    box(ax, 15, 75, 130, 50, "meme text")
    box(ax, 15, 195, 130, 50, "meme image")

    group(ax, 180, 22, 285, 270, "BACKBONES",
          "frozen; top 2 blocks opened in phase 2")
    box(ax, 198, 72, 250, 56, "CLIP text encoder", "77 tokens, 512-d")
    box(ax, 198, 192, 250, 56, "CLIP-ViT-B/32", "50 patches, 768-d")

    group(ax, 515, 22, 575, 285, "FUSION  ·  CA-AGFN",
          "both sides projected to 512 by CLIP's own projections, "
          "then trained")
    box(ax, 535, 75, 250, 56, "CrossModalAttention",
        "each side reads the other, per-head weights kept")
    box(ax, 535, 162, 250, 50, "SemanticClash",
        "|t − v| concat t · v, normalised pooled vectors")
    box(ax, 825, 75, 245, 56, "attention entropy",
        "per head, real tokens only, ÷ log n", edge=ORANGE, weight=1.8)
    box(ax, 825, 162, 245, 50, "AdaptiveGatedFusion",
        "fused = α·t + (1 − α)·v", edge=BLUE, weight=1.8)
    box(ax, 825, 240, 245, 50, "classifier head",
        "[fused ; clash] → MLP → one logit")
    box(ax, 1120, 240, 115, 50, "P(hateful)", edge=BLUE, weight=1.8)

    arrow(ax, (145, 100), (198, 100))
    arrow(ax, (145, 220), (198, 220))
    arrow(ax, (448, 94), (535, 96), "tokens", offset=(0, -9))
    arrow(ax, (448, 208), (535, 118))
    arrow(ax, (448, 114), (535, 175), dashed=True)
    arrow(ax, (448, 230), (535, 198), "pooled", dashed=True, offset=(0, 12))
    arrow(ax, (785, 103), (825, 103))
    arrow(ax, (785, 124), (825, 170), "t, v", offset=(-10, -6), align="right")
    arrow(ax, (947, 131), (947, 162), "entropy", offset=(6, 0), align="left")
    arrow(ax, (785, 187), (825, 187))
    arrow(ax, (700, 212), (825, 258), "clash", offset=(-12, 6), align="right")
    arrow(ax, (947, 212), (947, 240), "fused", offset=(6, 0), align="left")
    arrow(ax, (1070, 265), (1120, 265))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, facecolor=SURFACE)
    plt.close(figure)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "docs" / "pipeline.svg"))
    args = parser.parse_args()
    print(f"Wrote {draw(args.out)}")


if __name__ == "__main__":
    main()
