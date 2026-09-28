# CA-AGFN: Clash-Aware Adaptive Gated Fusion

[![tests](https://github.com/no0e/hateful-memes-ca-agfn/actions/workflows/tests.yml/badge.svg)](https://github.com/no0e/hateful-memes-ca-agfn/actions/workflows/tests.yml)
[![licence](https://img.shields.io/badge/licence-MIT-green)](LICENSE)

Hateful meme detection on Meta's Hateful Memes benchmark. A meme is often
hateful only through the combination of its text and its image, so the model
computes the *clash* between CLIP's text and image embeddings and gates how
much to trust each side.

<p align="center">
  <img src="docs/pipeline.svg" width="100%" alt="CLIP encodes the text and the image; a clash vector, cross-modal attention and an entropy-conditioned gate feed a classifier">
</p>

## Results

Test AUROC on the 500-meme dev split, held out entirely (checkpoints and
thresholds are chosen on 500 memes from train), mean ± sd over three seeds:

| Model | Test AUROC |
|---|---|
| CA-AGFN, full | 0.707 ± 0.006 |
| Concatenated CLIP embeddings, no fusion | **0.720 ± 0.004** |
| Concatenation + clash | 0.708 ± 0.005 |
| XLM-RoBERTa text encoder instead of CLIP | 0.657 ± 0.002 |
| Text only | 0.635 ± 0.003 |
| Image only | 0.655 ± 0.004 |

On the same split, the benchmark paper reports 0.651 for text-only BERT and
0.741 for Visual BERT.

What the ablations show:

- **A shared embedding space is what matters.** CLIP on both sides beats an
  XLM-RoBERTa text encoder by 0.05, and with XLM-R, shuffling the texts between
  memes leaves the score unchanged: that model ignores its text.
- **The fusion does not beat plain concatenation**, and adding the clash to the
  concatenation lowers the score.
- **The entropy gate reads a constant.** The text's attention over the image
  stays uniform: its logits vary by 0.034 across patches after training and
  0.033 at initialisation (`scripts/attention_diagnostics.py`).
- **The model uses both modalities.** Shuffling images between memes drops the
  AUROC to 0.641, shuffling texts to 0.615.

The 27 runs (9 variants × 3 seeds) are in [`results/`](results), summarised in
[`results/summary.md`](results/summary.md).

## Quick start

```bash
pip install -e ".[dev]"
pytest -q                          # 45 tests, CPU
python scripts/train.py --smoke    # the whole pipeline on generated data
```

## Trained weights

`weights/ca_agfn.safetensors`, stored with Git LFS (`git lfs pull`): the full
model, seed 0, test AUROC 0.712.

```python
from ca_agfn.checkpoint import load_checkpoint

model, meta = load_checkpoint("weights/ca_agfn.safetensors")
```

## Reproducing

```bash
pip install -e ".[data,figures]"
python scripts/fetch_data.py       # needs a Kaggle API token
python scripts/captions.py         # BLIP captions, once
python scripts/run_ablations.py    # 9 variants x 3 seeds, about 5 h on a T4
python scripts/aggregate.py        # results/summary.md
```

## Dataset

Meta's Hateful Memes: Kiela et al., *The Hateful Memes Challenge*, NeurIPS 2020,
[arXiv:2005.04790](https://arxiv.org/abs/2005.04790). It cannot be
redistributed: `scripts/fetch_data.py` downloads it, and no image or text from
it is in this repository.

## Licence

MIT for the code, see [LICENSE](LICENSE). The dataset is Meta's and carries its
own terms.
