"""Real optimizer updates and stage transitions for the fixed-potential VAE."""

import anndata
import lightning
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import scdiffeq as sdq
from scdiffeq.core.lightning_models import LightningSDE_VAE_FixedPotential


def snapshot(module):
    return {name: p.detach().clone() for name, p in module.named_parameters()}


def changed(before, after, prefix):
    return any(not torch.equal(p, after[name]) for name, p in before.items()
               if name.startswith(prefix))


@pytest.fixture
def network_kwargs():
    torch.manual_seed(7)
    return dict(
        latent_dim=3, encoder_n_hidden=1, decoder_n_hidden=1,
        encoder_dropout=0., decoder_dropout=0.,
        mu_hidden=[4], sigma_hidden=[4], mu_dropout=0., sigma_dropout=0.,
        pretrain_step_size=10, train_step_size=10, backend="tensorized",
    )


@pytest.fixture
def loader():
    generator = torch.Generator().manual_seed(7)
    return DataLoader(TensorDataset(
        torch.tensor([[0., 0.1]]).repeat(8, 1),
        torch.randn(8, 2, 6, generator=generator),
    ), batch_size=8)


@pytest.mark.parametrize("clip_algorithm", ["norm", "value"])
def test_pretraining_updates_autoencoder_only(tmp_path, network_kwargs, loader, clip_algorithm):
    model = LightningSDE_VAE_FixedPotential(
        data_dim=6, pretrain_epochs=1, **network_kwargs,
    )
    before = snapshot(model)
    model.hparams["gradient_clip_val"] = 0.5
    trainer = lightning.Trainer(
        accelerator="cpu", devices=1, max_epochs=1, limit_train_batches=1,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, default_root_dir=tmp_path,
        gradient_clip_algorithm=clip_algorithm,
    )
    trainer.fit(model, train_dataloaders=loader)
    after = snapshot(model)
    assert trainer.global_step == 1
    assert changed(before, after, "Encoder.")
    assert changed(before, after, "Decoder.")
    assert not changed(before, after, "DiffEq.")
    assert all(torch.isfinite(p).all() for p in after.values())
    assert torch.isfinite(trainer.callback_metrics["pretrain_rl_mse"])
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if clip_algorithm == "value":
        assert all(g.abs().max() <= .5 for g in gradients)
    else:
        assert torch.stack([g.norm() for g in gradients]).norm() <= .50001


class StageSnapshots(lightning.Callback):
    def __init__(self):
        self.epochs = []
        self.validation = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.stage = pl_module.PRETRAIN
        self.before = snapshot(pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        self.epochs.append((self.stage, self.before, snapshot(pl_module)))

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_before = snapshot(pl_module)
        self.steps_before = trainer.global_step

    def on_validation_epoch_end(self, trainer, pl_module):
        self.validation.append((
            self.val_before, snapshot(pl_module),
            self.steps_before, trainer.global_step,
        ))


@pytest.fixture
def public_model(tmp_path, monkeypatch, network_kwargs, request):
    monkeypatch.chdir(tmp_path)
    X = np.random.default_rng(7).normal(size=(40, 6)).astype(np.float32)
    adata = anndata.AnnData(X=X, obs=pd.DataFrame(
        {"t": np.repeat([0., .1], 20), "W": np.ones(40)},
        index=[str(i) for i in range(40)],
    ))
    adata.obsm["X_pca"] = X
    network_kwargs.update(getattr(request, "param", {}))
    # Keep the constructor's default pretrain_epochs: the public call must
    # propagate its requested stage lengths to the Lightning module.
    return sdq.scDiffEq(
        adata, velocity_ratio_params=None, batch_size=8, num_workers=0,
        working_dir=str(tmp_path), **network_kwargs,
    )


@pytest.mark.parametrize("routine", ["fit", "separate"])
@pytest.mark.parametrize("public_model", [
    {},
    {"pretrain_epochs": 1, "train_epochs": 1,
     "pretrain_step_size": None, "train_step_size": None},
], indirect=True, ids=["default_epochs", "one_epoch"])
def test_public_training_transitions_and_validates(public_model, routine):
    observer = StageSnapshots()
    options = dict(accelerator="cpu", devices=1,
                   limit_train_batches=1, limit_val_batches=1,
                   num_sanity_val_steps=0, enable_model_summary=False)
    if routine == "fit":
        public_model.fit(pretrain_epochs=1, train_epochs=1,
                         pretrain_callbacks=[observer], train_callbacks=[observer],
                         **options)
    else:
        public_model.pretrain(pretrain_epochs=1, pretrain_callbacks=[observer], **options)
        public_model.train(train_epochs=1, train_callbacks=[observer], **options)

    assert [stage for stage, _, _ in observer.epochs] == [True, False]
    _, before, pretrained = observer.epochs[0]
    assert changed(before, pretrained, "Encoder.")
    assert changed(before, pretrained, "Decoder.")
    assert not changed(before, pretrained, "DiffEq.")
    _, before, trained = observer.epochs[1]
    assert changed(before, trained, "DiffEq.")
    assert all(torch.isfinite(p).all() for p in trained.values())
    assert torch.isfinite(public_model.DiffEq.sinkhorn_loss).all()
    assert observer.validation
    for before, after, steps_before, steps_after in observer.validation:
        assert steps_before == steps_after
        assert all(torch.equal(p, after[name]) for name, p in before.items())
