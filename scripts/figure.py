"""Draw the results figure from the result files, with no model and no GPU.

    python scripts/figure.py

Four panels. The ablations, each a mean over seeds with the seeds shown. The
shuffle test: what happens to the AUROC when each text is paired with another
meme's image, or each image with another meme's text. The entropy gate's claim,
measured on every test meme. The ROC curve of the full model, one line per
seed.

No dataset image or text appears anywhere: the benchmark is licensed for
research and built around hateful content.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ca_agfn.config import ROOT  # noqa: E402
from ca_agfn.results import load, summarise  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_1 = "#2a78d6"   # blue
SERIES_2 = "#eb6834"   # orange
SERIES_3 = "#1a9e77"   # aqua

# Kiela et al. (2020), Table 1, validation AUROC. The same 500 memes that are
# the test split here.
PUBLISHED = (("Text BERT", 0.6505), ("MMBT-Grid", 0.6673),
             ("Visual BERT", 0.7414))

SHORT = {
    "full": "CA-AGFN, full",
    "no_entropy": "− entropy in the gate",
    "no_clash": "− clash vector",
    "no_captions": "− BLIP captions",
    "xlmr": "XLM-R text encoder",
    "concat": "Concat, no fusion",
    "concat_clash": "Concat + clash",
    "text_only": "Text only",
    "image_only": "Image only",
}


def reference(ax, x, name, colour=BASELINE):
    """A dashed vertical reference, labelled along itself at the top."""
    ax.axvline(x, color=colour, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    ax.annotate(name, (x, 1), xycoords=("data", "axes fraction"),
                xytext=(3, -4), textcoords="offset points", rotation=90,
                fontsize=7.5, color=INK_MUTED, ha="left", va="top",
                bbox={"boxstyle": "square,pad=0.1", "fc": SURFACE, "ec": "none"},
                zorder=5)


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "sans-serif"], "font.size": 9,
        "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "axes.titlelocation": "left",
        "axes.labelsize": 9, "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.8, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.7, "xtick.color": INK_MUTED,
        "ytick.color": INK_SECONDARY, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "legend.frameon": False,
        "legend.fontsize": 8.5,
    })


def strip(ax, keep=("bottom",)):
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def panel_ablations(ax, runs, summary):
    order = [v for v in SHORT if v in summary][::-1]
    for row, variant in enumerate(order):
        scores = [run["test"]["auroc"] for run in runs[variant]]
        stat = summary[variant]["test_auroc"]
        colour = SERIES_1 if variant == "full" else INK_SECONDARY
        ax.hlines(row, stat["mean"] - stat["std"], stat["mean"] + stat["std"],
                  color=colour, linewidth=2, zorder=3)
        ax.scatter(scores, [row] * len(scores), s=16, color=colour, alpha=0.35,
                   edgecolor="none", zorder=3)
        ax.scatter([stat["mean"]], [row], s=64, color=colour, zorder=4,
                   edgecolor=SURFACE, linewidth=2)
        ax.annotate(f"{stat['mean']:.3f}", (stat["mean"] + stat["std"], row),
                    textcoords="offset points", xytext=(6, -3), fontsize=8.5,
                    color=INK, fontweight="bold" if variant == "full" else None)

    for name, value in PUBLISHED:
        reference(ax, value, name)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHORT[v] for v in order])
    ax.set_ylim(-0.6, len(order) - 0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Test AUROC, mean ± sd over seeds (dots are seeds); dashed: "
                  "published baselines")
    ax.set_title("What each part is worth")
    strip(ax)


def panel_shuffle(ax, summary):
    order = [v for v in ("image_only", "text_only", "xlmr", "concat",
                         "concat_clash", "full") if v in summary]
    conditions = (("test_auroc", "Intact", SERIES_1),
                  ("auroc_shuffled_image", "Images shuffled", SERIES_2),
                  ("auroc_shuffled_text", "Texts shuffled", SERIES_3))
    for row, variant in enumerate(order):
        values = [summary[variant][key]["mean"] for key, _, _ in conditions]
        ax.hlines(row, min(values), max(values), color=BASELINE, linewidth=1.2,
                  zorder=2)
        for (key, name, colour), value in zip(conditions, values, strict=True):
            label = name if row == 0 else None
            if key == "test_auroc":
                # A ring on top, so a shuffle that changes nothing shows as a
                # dot inside the ring instead of hiding the intact score.
                ax.scatter([value], [row], s=130, facecolors="none",
                           edgecolors=colour, linewidth=2, zorder=5,
                           label=label)
            else:
                ax.scatter([value], [row], s=64, color=colour, zorder=4,
                           edgecolor=SURFACE, linewidth=2, label=label)

    reference(ax, 0.5, "chance")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHORT[v] for v in order])
    ax.set_ylim(-0.6, len(order) - 0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Test AUROC, mean over seeds")
    ax.set_title("Does the model read both halves of the meme?")
    ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, -0.13),
              handletextpad=0.2, columnspacing=1.2)
    strip(ax)


def panel_gate(ax, runs):
    run = runs["full"][0]
    entropy = np.array(run["predictions"]["entropy"])
    gate = np.array(run["predictions"]["gate"])
    ax.scatter(entropy, gate, s=14, color=SERIES_1, alpha=0.35,
               edgecolor="none", zorder=2)

    order = np.argsort(entropy)
    window = max(15, len(entropy) // 20)
    smoothed = np.convolve(gate[order], np.ones(window) / window, mode="valid")
    ax.plot(entropy[order][window - 1:], smoothed, color=SERIES_2,
            linewidth=2, zorder=4, label="Rolling mean")

    correlations = [r["gate"]["r_entropy"] for r in runs["full"]]
    text = f"r = {correlations[0]:+.2f} (seed {run['seed']})"
    if len(correlations) > 1:
        text += (f"\nr = {np.mean(correlations):+.2f} ± "
                 f"{np.std(correlations, ddof=1):.2f} over "
                 f"{len(correlations)} seeds")
    ax.annotate(text, (0.03, 0.05), xycoords="axes fraction", fontsize=9,
                color=INK, fontweight="bold")
    ax.set_xlabel("Entropy of the text's attention over the image, "
                  "real tokens only")
    ax.set_ylabel("Gate: weight given to the text")
    ax.set_title("Does the entropy gate do what it claims?")
    ax.legend(loc="upper right")
    strip(ax, keep=("left", "bottom"))


def panel_roc(ax, runs, summary):
    ax.plot([0, 1], [0, 1], color=BASELINE, linewidth=1.2,
            linestyle=(0, (3, 3)), zorder=2, label="Chance")
    for index, run in enumerate(runs["full"]):
        ax.plot(run["roc"]["fpr"], run["roc"]["tpr"], color=SERIES_1,
                linewidth=1.6, alpha=0.8, zorder=3,
                label="CA-AGFN, one line per seed" if index == 0 else None)
    stat = summary["full"]["test_auroc"]
    ax.annotate(f"AUROC {stat['mean']:.3f} ± {stat['std']:.3f}", (0.97, 0.2),
                xycoords="axes fraction", ha="right", fontsize=9, color=INK,
                fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Test ROC, full model")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.26))
    strip(ax, keep=("left", "bottom"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--out", default=str(ROOT / "docs" / "results.png"))
    args = parser.parse_args()

    runs = load(args.results)
    if "full" not in runs:
        raise SystemExit(f"No results for the full model under {args.results}.")
    summary = summarise(runs)

    style()
    figure, axes = plt.subplots(2, 2, figsize=(13, 9.4),
                                gridspec_kw={"hspace": 0.45, "wspace": 0.28})
    panel_ablations(axes[0, 0], runs, summary)
    panel_shuffle(axes[0, 1], summary)
    panel_gate(axes[1, 0], runs)
    panel_roc(axes[1, 1], runs, summary)

    figure.suptitle("CA-AGFN on the Hateful Memes dev split, held out as test",
                    fontsize=13, fontweight="bold", color=INK, x=0.012,
                    ha="left", y=0.985)
    figure.text(
        0.012, 0.95,
        "Checkpoints and thresholds chosen on 500 memes held out from train; "
        "the 500 test memes are scored once. No dataset image or text is "
        "reproduced.",
        fontsize=8.5, color=INK_MUTED, ha="left")
    figure.subplots_adjust(left=0.15, right=0.98, top=0.88, bottom=0.07)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
