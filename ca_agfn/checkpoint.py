"""Save and load a trained model as its tuned weights plus its config.

Only the parameters training changed are written, in safetensors: about
100 MB, where the first version pickled the whole 1.4 GB state dict, frozen
backbones included. The backbones come back from `from_pretrained` as usual.
safetensors also means loading a checkpoint cannot execute code, which
`torch.load(weights_only=False)` could.
"""
import dataclasses
import json
from pathlib import Path

from safetensors.torch import load_file, save_file

from .config import Config
from .model import CAAGFN


def save_checkpoint(model, path, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.contiguous()
             for name, tensor in model.tuned_state_dict().items()}
    save_file(state, str(path))
    meta = {"config": model.config.to_dict(), **(extra or {})}
    path.with_suffix(".json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    return path


def load_checkpoint(path, device="cpu"):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    known = {field.name for field in dataclasses.fields(Config)}
    config = Config(**{key: value for key, value in meta["config"].items()
                       if key in known})
    model = CAAGFN(config)
    model.load_tuned_state_dict(load_file(str(path)))
    return model.to(device).eval(), meta
