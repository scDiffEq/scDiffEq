# tests/test_device_configuration.py

"""Guards for how the Lightning Trainer picks its devices.

The trainer config used to overwrite whatever ``devices`` the caller asked for
with ``torch.cuda.device_count()``. On a multi-GPU machine that turned every run
into DDP, and in a notebook Lightning launches DDP by forking the kernel, which
cannot re-initialize CUDA -- the crash reported in #111.
"""

# -- import packages: ---------------------------------------------------------
import logging
import pytest
import torch

# -- import local dependencies: -----------------------------------------------
from scdiffeq.core.configs._lightning_trainer_configuration import (
    LightningTrainerConfiguration,
)
from scdiffeq.core.configs._progress_bar_config import ProgressBarConfig


# -- fixtures: ----------------------------------------------------------------
@pytest.fixture
def config_factory(tmp_path, monkeypatch):
    """Build a trainer config that believes it is in ``env`` with ``n_gpu`` GPUs."""

    def _factory(env: str, n_gpu: int):
        monkeypatch.setattr(torch.cuda, "device_count", lambda: n_gpu)
        monkeypatch.setattr(ProgressBarConfig, "_detect_env", lambda self: env)
        config = LightningTrainerConfiguration(save_dir=str(tmp_path / "model"))
        config._progress_bar_config = ProgressBarConfig(total_epochs=1)
        return config

    return _factory


# -- tests: -------------------------------------------------------------------
@pytest.mark.parametrize("devices", [1, 2, [0, 2], "auto", None])
def test_notebook_never_exceeds_one_device(config_factory, devices):
    """Multi-device training from a notebook kernel is the #111 crash."""
    config = config_factory(env="jupyter", n_gpu=4)

    assert config._resolve_devices(devices) == 1


def test_notebook_downgrade_is_announced(config_factory, caplog):
    """Silently training on fewer devices than asked for is its own bug."""
    config = config_factory(env="jupyter", n_gpu=4)

    with caplog.at_level(logging.WARNING):
        config._resolve_devices(4)

    assert "using 1" in caplog.text


def test_colab_is_treated_as_a_notebook(config_factory):
    """Colab runs the same fork-based launcher as Jupyter."""
    config = config_factory(env="colab", n_gpu=2)

    assert config._resolve_devices(None) == 1


@pytest.mark.parametrize("devices", [1, 2, [0, 2]])
def test_explicit_devices_are_honored(config_factory, devices):
    """The requested count reached Lightning as-is, not as the GPU count."""
    config = config_factory(env="terminal", n_gpu=4)

    assert config._resolve_devices(devices) == devices


@pytest.mark.parametrize("devices", [None, "auto", -1])
def test_auto_uses_every_gpu_outside_a_notebook(config_factory, devices):
    """The pre-existing default -- every visible CUDA device -- is preserved."""
    config = config_factory(env="terminal", n_gpu=4)

    assert config._resolve_devices(devices) == 4


def test_auto_without_cuda_is_one_device(config_factory):
    """``devices=None`` reached the CPU accelerator and raised TypeError."""
    config = config_factory(env="terminal", n_gpu=0)

    assert config._resolve_devices(None) == 1


def test_requested_devices_reach_the_trainer(config_factory):
    """End to end: the Trainer this config builds honors the request."""
    config = config_factory(env="jupyter", n_gpu=4)
    trainer = config(logger=False, max_epochs=1, accelerator="cpu", devices=1)

    assert trainer.num_devices == 1
    assert type(trainer.strategy).__name__ == "SingleDeviceStrategy"
