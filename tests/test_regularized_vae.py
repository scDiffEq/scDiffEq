"""Selection, latent regularization, and persistence of the combined VAE/SDE."""

import anndata
import lightning
import numpy as np
import pandas as pd
import pytest
import torch
import torchsde

import scdiffeq as sdq
from scdiffeq.core import lightning_models
from scdiffeq.core.configs import LightningModelConfiguration
from scdiffeq.core.configs._lightning_callbacks_configuration import InterTrainerEpochCounter
from .test_vae_training import network_kwargs, loader, StageSnapshots, snapshot, changed


MODEL_NAME = "LightningSDE_VAE_FixedPotential_RegularizedVelocityRatio"


@pytest.fixture(params=[1, 2], ids=["one_brownian_dim", "two_brownian_dims"])
def model(network_kwargs, request):
    cls = getattr(lightning_models, MODEL_NAME)
    return cls(data_dim=6, pretrain_epochs=1, brownian_dim=request.param,
               velocity_ratio_params={"target": 2.5, "enforce": 1., "method": "square"},
               **network_kwargs)


@pytest.mark.parametrize("width,expected", [
    (50, "LightningSDE_FixedPotential_RegularizedVelocityRatio"),
    (51, MODEL_NAME),
    (60, MODEL_NAME),
])
def test_default_model_selection(tmp_path, monkeypatch, width, expected):
    monkeypatch.chdir(tmp_path)
    X = np.random.default_rng(7).normal(size=(90, width)).astype(np.float32)
    adata = anndata.AnnData(X=X, obs=pd.DataFrame(
        {"t": np.repeat([0., .1, .2], 30), "W": np.ones(90)},
        index=[str(i) for i in range(90)],
    ))
    adata.obsm["X_pca"] = X
    model = sdq.scDiffEq(adata)
    assert type(model.DiffEq).__name__ == expected
    assert model.DiffEq.hparams.latent_dim == 50
    assert model.DiffEq.hparams.velocity_ratio_params["enforce"] == 100


def test_regularizer_contributes_latent_gradients(model, loader):
    batch = next(iter(loader))
    model.eval()
    predictions = model(batch[1][:, 0], batch[0][0])
    assert predictions.shape == (2, 8, 6)
    model.step(batch, 0, stage="validation")
    assert model.sinkhorn_loss.shape == model.velocity_ratio_loss.shape == (2,)
    assert torch.isfinite(model.total_loss).all()
    torch.testing.assert_close(model.total_loss,
                               model.sinkhorn_loss + model.velocity_ratio_loss)
    for network in (model.DiffEq.mu, model.DiffEq.sigma):
        gradients = torch.autograd.grad(
            model.velocity_ratio_loss.sum(), list(network.parameters()),
            allow_unused=True, retain_graph=True,
        )
        assert any(g is not None and g.abs().sum() > 0 for g in gradients)
        assert all(torch.isfinite(g).all() for g in gradients if g is not None)


def test_training_validation_and_checkpoint(model, loader, tmp_path):
    observer = StageSnapshots()
    trainer = lightning.Trainer(
        accelerator="cpu", devices=1, max_epochs=2, limit_train_batches=1,
        limit_val_batches=1, num_sanity_val_steps=0, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, default_root_dir=tmp_path,
        callbacks=[observer, InterTrainerEpochCounter()],
    )
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
    assert [stage for stage, _, _ in observer.epochs] == [True, False]
    _, initial, pretrained = observer.epochs[0]
    assert changed(initial, pretrained, "Encoder.")
    assert changed(initial, pretrained, "Decoder.")
    assert not changed(initial, pretrained, "DiffEq.")
    _, _, trained = observer.epochs[1]
    assert changed(pretrained, trained, "DiffEq.")
    assert trainer.global_step == 2
    assert torch.isfinite(model.sinkhorn_loss).all()
    assert torch.isfinite(model.velocity_ratio_loss).all()
    for before, after, steps_before, steps_after in observer.validation:
        assert steps_before == steps_after
        assert all(torch.equal(p, after[name]) for name, p in before.items())

    path = tmp_path / "model.ckpt"
    trainer.save_checkpoint(path)
    restored = type(model).load_from_checkpoint(path, weights_only=False)
    assert restored.hparams.data_dim == 6
    assert restored.hparams.latent_dim == 3
    assert restored.hparams.velocity_ratio_params == model.hparams.velocity_ratio_params
    assert not restored.PRETRAIN
    model.eval()
    restored.eval()
    batch = next(iter(loader))
    def brownian():
        return torchsde.BrownianInterval(
            t0=0., t1=.1, size=(8, model.hparams.brownian_dim), entropy=123,
        )
    expected = model(batch[1][:, 0], batch[0][0], bm=brownian())
    actual = restored(batch[1][:, 0], batch[0][0], bm=brownian())
    torch.testing.assert_close(actual, expected)


def test_unsupported_combination_explains_options():
    config = LightningModelConfiguration(
        data_dim=60, latent_dim=50, potential_type="prior",
        velocity_ratio_params={"enforce": 100},
    )
    with pytest.raises(ValueError) as error:
        config({})
    message = str(error.value)
    assert "data_dim=60" in message
    assert "latent_dim=50" in message
    assert "VAE" in message
    assert "velocity_ratio_params=None" in message
    assert "LightningSDE_VAE_PriorPotential" in message


@pytest.mark.parametrize("diffusion", [0., 1.])
def test_zero_enforcement_keeps_combined_model(network_kwargs, diffusion):
    config = LightningModelConfiguration(
        data_dim=6, latent_dim=3,
        velocity_ratio_params={"target": 2.5, "enforce": 0, "method": "square"},
    )
    model = config(dict(data_dim=6, coef_diffusion=diffusion,
                        velocity_ratio_params=config._velocity_ratio_params,
                        **network_kwargs))
    assert type(model).__name__ == MODEL_NAME
    loss = model._velocity_ratio_loss(torch.randn(2, 8, 3))
    torch.testing.assert_close(loss, torch.zeros(2))
