# CA-AGFN: Clash-Aware Adaptive Gated Fusion

Hateful meme detection on Meta's Hateful Memes benchmark, built on one premise:
a meme is rarely hateful in its text alone or its image alone, but in the gap
between them. The model computes that gap explicitly, a *clash* vector between
CLIP's text and image embeddings, and gates how much to trust each side.

```bash
pip install -e ".[dev]"
python scripts/train.py --smoke    # the whole pipeline on generated data
```

MIT for the code, see [LICENSE](LICENSE). The dataset is Meta's and carries its
own terms.
