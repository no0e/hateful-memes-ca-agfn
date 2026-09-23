"""Fetch the Hateful Memes dataset into data/.

    python scripts/fetch_data.py

The dataset is not redistributable, so this downloads it rather than shipping
it. It needs credentials, which is the one step this script cannot do for you.
Either works:

    ~/.kaggle/kaggle.json        the classic file, with a username and a key
    KAGGLE_API_TOKEN=KGAT_...    the newer prefixed token, in the environment

Kaggle now issues the second kind from Settings, API, Create New Token, and a
token of that shape in a file called anything else is not picked up by
kagglehub. The alternative source is Meta's own release at hatefulmemes.org,
which requires agreeing to their research licence.

What lands in data/: train.jsonl, dev.jsonl, test.jsonl and an img/ directory of
about 3.4 GB.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASET = "parthplc/facebook-hateful-meme-dataset"
EXPECTED = ("train.jsonl", "dev.jsonl")


def already_there(target):
    return all((target / name).exists() for name in EXPECTED) \
        and (target / "img").exists()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "data"))
    args = parser.parse_args()

    target = Path(args.out)
    if already_there(target):
        print(f"{target} already has the dataset. Nothing to do.")
        return

    try:
        import kagglehub
    except ImportError:
        raise SystemExit(
            "kagglehub is not installed. pip install kagglehub, then run this "
            "again."
        )

    credentials = Path.home() / ".kaggle" / "kaggle.json"
    if not credentials.exists() and not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit(
            f"No Kaggle credentials: neither {credentials} nor a\n"
            "KAGGLE_API_TOKEN in the environment.\n\n"
            "On Kaggle: Settings, API, Create New Token. If that hands you a\n"
            "kaggle.json:\n\n"
            "    mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/\n"
            "    chmod 600 ~/.kaggle/kaggle.json\n\n"
            "If it hands you a token beginning KGAT_, export it instead:\n\n"
            "    export KAGGLE_API_TOKEN='KGAT_...'\n\n"
            "A KGAT_ token written to a file is not read by kagglehub; it has\n"
            "to be in the environment. This script never reads or transmits\n"
            "either credential itself, the Kaggle client does."
        )

    print(f"Downloading {DATASET} ...")
    downloaded = Path(kagglehub.dataset_download(DATASET))

    # The mirror nests everything one level down under data/.
    source = downloaded / "data" if (downloaded / "data").exists() else downloaded
    target.mkdir(parents=True, exist_ok=True)

    for item in source.iterdir():
        destination = target / item.name
        if destination.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)
        print(f"  {item.name}")

    if not already_there(target):
        raise SystemExit(
            f"The download finished but {target} is still missing one of "
            f"{EXPECTED} or img/. Check what landed in {source}."
        )

    images = len(list((target / "img").glob("*")))
    print(f"\nReady: {target}, {images:,} images.")


if __name__ == "__main__":
    main()
