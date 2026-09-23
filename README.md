# CA-AGFN: Clash-Aware Adaptive Gated Fusion

[![tests](https://github.com/no0e/hateful-memes-ca-agfn/actions/workflows/tests.yml/badge.svg)](https://github.com/no0e/hateful-memes-ca-agfn/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![licence](https://img.shields.io/badge/licence-MIT-green)](LICENSE)

Detecting hateful memes, on the premise that a meme is rarely hateful in its
text alone or its image alone. It is hateful in the gap between them: a caption
that is innocuous under one picture and vicious under another.

Meta's Hateful Memes benchmark is built to force that point. For most hateful
memes in it there exists a benign one carrying the same text over a different
image, or the same image under different text. A model that reads one modality
and ignores the other cannot separate those pairs even in principle.

So this model builds the gap explicitly rather than hoping a classifier finds it
in a concatenation, and then decides how much to trust each side.

## The architecture

**Cross-modal attention.** Each modality reads the other, and the text-to-vision
attention distribution is kept, because the gate downstream is computed from it.

**Semantic clash.** The gap itself, as a vector: `|t - v|` concatenated with
`t * v`, projected down. The first term is how far apart the two pooled
representations are, the second is where they agree. A plain concatenation of
`t` and `v` contains the same information and makes none of it explicit.

**Language-conditioned entropy gate.** This is the part that is mine. The
entropy of the text's attention over the image says how sharply the text is
pointing at anything. Text that commits to a region is text worth weighting.
Text whose attention is spread evenly over every patch is text that is not
saying much, and the image should decide instead. The gate reads that entropy
alongside the clash, a second gate reads the image, and a learned scalar
interpolates between the two strategies rather than committing to one at design
time.

```
alpha_text   = sigmoid(W [t ; clash ; entropy])
alpha_vision = sigmoid(W [v ; clash])
alpha        = lambda * alpha_text + (1 - lambda) * (1 - alpha_vision)
fused        = alpha * t + (1 - alpha) * v
```

Entropy is normalised by `log(n_patches)`, so it lands in [0, 1] and keeps its
meaning across image encoders with different patch counts. It is also
renormalised before it is measured, because attention dropout in training mode
leaves weights that sum to about 0.9 rather than to 1, and an entropy taken over
that is being read off something that is not a distribution.

Backbones are XLM-RoBERTa and CLIP-ViT-B/32. Memes carry multilingual and
transliterated text, which is what XLM-R is for.

## The bug this repository exists to fix

An earlier version of this work trained phase one to an AUROC of 0.611 and then
died four lines into phase two, on ten consecutive non-finite losses. The
diagnosis is worth writing down, because the bug was in the safety net.

That loop checked whether the **loss** was finite and skipped the step if it was
not. At the first step of phase two the loss was finite. The backward pass
through the newly unfrozen backbone produced non-finite **gradients**.
`clip_grad_norm_` computed a total norm of NaN, divided every gradient by it,
and so turned all of them into NaN. The optimiser wrote those into the weights.
From the next step onward every forward pass returned NaN, the loss check fired
on all of them, and the run aborted ten steps later having lost the model at
step zero.

The fix is one check in the right place: gradients are tested for finiteness
after the backward pass and before the clip. `tests/test_model.py` demonstrates
the mechanism rather than asserting it, by clipping a set of gradients where
exactly one entry is NaN and showing that an untouched, finite gradient
elsewhere in the model comes back non-finite.

```python
if not gradients_are_finite(model):
    skipped += 1
    optimiser.zero_grad(set_to_none=True)
    continue
torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
optimiser.step()
```

`skipped_steps` is recorded every epoch and printed at the end, because a guard
that silently eats half the batches is its own kind of failure.

## Quick start

```bash
git clone https://github.com/no0e/hateful-memes-ca-agfn.git
cd hateful-memes-ca-agfn
pip install -r requirements.txt

pytest tests/ -q                  # 11 tests, no dataset and no GPU needed
python scripts/train.py --smoke   # 64 samples, one epoch per phase
```

The smoke run exercises the whole path on a CPU in a couple of minutes. It
proves the code runs. It proves nothing about the model.

For a real run you need the dataset:

```bash
python scripts/fetch_data.py      # needs a Kaggle token, see below
python scripts/captions.py        # optional, one pass of BLIP over the images
python scripts/train.py
```

## The dataset

Meta's Hateful Memes: 10,000 memes, each an image with its overlaid text already
extracted, labelled hateful or not, about 36% positive. It is not
redistributable, so `scripts/fetch_data.py` downloads it and this repository
ships none of it.

It needs a Kaggle API token at `~/.kaggle/kaggle.json`, or Meta's own release at
hatefulmemes.org under their research licence. The script tells you which step
is missing and never handles the token itself.

`scripts/captions.py` is optional. It runs BLIP once over every image and writes
a caption beside each row, so the text encoder reads
`<overlaid text> [SEP] <what the picture shows>` and the clash feature has two
descriptions of the same meme to compare. Training picks the captioned files up
automatically if they exist.

## Training

Two phases, for a reason.

Phase one trains only the new modules against frozen backbones. Starting with
everything unfrozen sends a large gradient from a randomly initialised head
straight into pretrained weights, which is how the useful part of a pretrained
encoder is destroyed in the first few steps.

Phase two opens the top two transformer blocks of each backbone, with layer-wise
learning rate decay so deeper blocks move less. Embeddings, position tables and
final layer norms stay frozen throughout.

Both phases validate on EMA weights, keep the best checkpoint by AUROC, and stop
early when it stops improving.

## Layout

```
ca_agfn/
  config.py      every hyperparameter, with the reason beside the chosen ones
  data.py        the dataset, tokenised once, with a missing-image count
  model.py       cross-modal attention, semantic clash, the entropy gate
  training.py    two-phase training, EMA, and the gradient guard
scripts/         fetch_data, captions, train
tests/           11 tests, CPU only, seconds to run
```

## Requirements

Python 3.10 or later, PyTorch 2.1 or later, transformers, scikit-learn, pandas,
Pillow. A GPU for training; the tests and the smoke run do not need one.

## License

MIT for the code. See [LICENSE](LICENSE). The dataset is Meta's and carries its
own terms.
