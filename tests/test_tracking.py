"""Experiment tracking: silent when off, one point per epoch when on."""
import pytest

from ca_agfn.tracking import Tracker


def test_a_disabled_tracker_imports_nothing_and_logs_nothing():
    tracker = Tracker(enabled=False)
    assert not tracker.enabled
    tracker.epoch(1, 1, loss=0.5)
    tracker.summary({"auroc": 0.7})
    tracker.finish()


def test_an_enabled_tracker_logs_epochs_in_global_order(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")

    tracker = Tracker(enabled=True, project="ca-agfn-test", name="unit",
                      config={"variant": "full"})
    logged = []
    original = tracker.run.log
    tracker.run.log = lambda values, **kw: (logged.append(values),
                                            original(values, **kw))
    tracker.epoch(1, 1, loss=0.9, val_auroc=0.60)
    tracker.epoch(1, 2, loss=0.8, val_auroc=0.62)
    tracker.epoch(2, 1, loss=0.7, val_auroc=0.65)
    tracker.summary({"auroc": 0.70, "ci95": [0.6, 0.8]}, prefix="test/")
    assert [point["global_epoch"] for point in logged] == [1, 2, 3]
    assert [point["phase"] for point in logged] == [1, 1, 2]
    assert tracker.run.summary["test/auroc"] == 0.70
    assert "test/ci95" not in tracker.run.summary  # only scalars
    tracker.finish()
