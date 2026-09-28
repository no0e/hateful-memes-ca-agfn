"""The splits, the captions and the augmentation."""
import numpy as np
import pandas as pd
from PIL import Image

from ca_agfn.data import augment, captions_available, hold_out, split_file


def test_the_loader_prefers_the_captioned_split(tmp_path):
    """captions.py writes <split>_captioned.jsonl. If the loader does not look
    for it, an expensive BLIP pass changes nothing and nothing says so."""
    (tmp_path / "train.jsonl").write_text("{}", encoding="utf-8")
    assert split_file(tmp_path, "train").name == "train.jsonl"

    (tmp_path / "train_captioned.jsonl").write_text("{}", encoding="utf-8")
    assert split_file(tmp_path, "train").name == "train_captioned.jsonl"


def test_captions_can_be_turned_off(tmp_path):
    (tmp_path / "dev.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "dev_captioned.jsonl").write_text("{}", encoding="utf-8")
    assert split_file(tmp_path, "dev", use_captions=False).name == "dev.jsonl"


def test_captions_count_only_when_both_splits_have_them(tmp_path):
    """Captioned training text and bare test text would be a distribution
    shift introduced by the file system."""
    (tmp_path / "train_captioned.jsonl").write_text("{}", encoding="utf-8")
    assert not captions_available(tmp_path)
    (tmp_path / "dev_captioned.jsonl").write_text("{}", encoding="utf-8")
    assert captions_available(tmp_path)


def test_the_hold_out_is_stratified_and_the_same_every_time():
    frame = pd.DataFrame({"id": range(1000),
                          "label": [1] * 360 + [0] * 640})
    train, val = hold_out(frame, 100, seed=0)
    again_train, again_val = hold_out(frame, 100, seed=0)

    assert len(val) == 100 and len(train) == 900
    assert val["label"].sum() == 36
    assert set(val["id"]).isdisjoint(train["id"])
    assert list(val["id"]) == list(again_val["id"])


def test_augmentation_never_mirrors_the_image():
    """A flip mirrors the meme's text, which is part of what the model reads."""
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    pixels[:, 32:] = 255
    image = Image.fromarray(pixels)

    for _ in range(50):
        out = np.asarray(augment(image)).astype(float)
        assert out[:, :24].mean() < out[:, 40:].mean()
