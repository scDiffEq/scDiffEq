"""Auxiliary observations must not alter growth weights or the objective."""

import anndata
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

import scdiffeq as sdq
from scdiffeq.core.lightning_models.base import BatchProcessor


@pytest.fixture
def observations():
    X = np.random.default_rng(7).normal(size=(40, 3)).astype('float32')
    obs = pd.DataFrame({
        't': np.repeat([0., .1], 20),
        'W': np.tile(np.linspace(-1, 1, 20), 2),
        'growth': np.tile(np.linspace(-1, 1, 20), 2),
        'label': np.tile([10., 20.], 20),
        'other': np.arange(40, dtype=float),
    }, index=[str(i) for i in range(40)])
    adata = anndata.AnnData(X, obs=obs)
    adata.obsm['X_pca'] = X.copy()
    return adata


def make_model(adata, obs_keys, weight_key, tmp_path):
    np.random.seed(17)
    torch.manual_seed(17)
    return sdq.scDiffEq(
        adata.copy(), obs_keys=obs_keys, weight_key=weight_key,
        latent_dim=3, use_key='X_pca', time_key='t', num_workers=0,
        mu_hidden=[4], sigma_hidden=[4], mu_dropout=0., sigma_dropout=0.,
        potential_type=None, velocity_ratio_params=None,
        backend='tensorized', working_dir=str(tmp_path), silent=True,
    )


@pytest.mark.parametrize('weight_key', ['W', 'growth'])
@pytest.mark.parametrize('keys', [[], ['label'], ['label', 'other'],
                                 ['other', 'label'], ['label', 'WEIGHT', 'other'],
                                 ['WEIGHT', 'label'], ['label', 'WEIGHT', 'WEIGHT']])
def test_metadata_preserves_weights_and_training_loss(observations, tmp_path, weight_key, keys):
    keys = [weight_key if k == 'WEIGHT' else k for k in keys]
    original = keys.copy()
    baseline = make_model(observations, [], weight_key, tmp_path)
    augmented = make_model(observations, keys, weight_key, tmp_path)
    assert keys == original
    batches = []
    for model in (baseline, augmented):
        np.random.seed(19)
        torch.manual_seed(19)
        batches.append(next(iter(DataLoader(model.LitDataModule.train_dataset, batch_size=8))))
    torch.testing.assert_close(batches[0][1], batches[1][1])
    raw = batches[0][2].clone()
    expected = torch.softmax(.1 * raw[:, 0, :], dim=0)
    for batch in batches:
        processed = BatchProcessor(batch, 0)
        torch.testing.assert_close(processed.W_hat[0], torch.full((8, 1), 1 / 8))
        torch.testing.assert_close(processed.W_hat[1], expected)
        assert processed.F_idx is None
    losses = []
    for model, batch in zip((baseline, augmented), batches):
        np.random.seed(42)
        torch.manual_seed(42)
        losses.append(model.DiffEq.training_step(batch, 0))
    assert torch.isfinite(losses[0])
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)


def test_weight_processing_does_not_modify_input():
    batch = [torch.tensor([[0., 1.], [0., 1.]]), torch.zeros(2, 2, 3),
             torch.tensor([[[0.], [0.]], [[1.], [1.]]])]
    raw = batch[2].clone()
    first = BatchProcessor(batch, 0).W_hat
    torch.testing.assert_close(batch[2], raw)
    torch.testing.assert_close(BatchProcessor(batch, 0).W_hat, first)


def test_explicit_fate_contract():
    batch = [torch.tensor([[0., 1.], [0., 1.]]), torch.zeros(2, 2, 3),
             torch.ones(2, 2, 1), torch.tensor([[[12.], [13.]], [[21.], [22.]]])]
    assert BatchProcessor(batch, 0).F_idx is None
    assert BatchProcessor(batch, 0, fate_bias=True).F_idx.tolist() == ['12', '21']


@pytest.mark.parametrize('kind', ['missing', 'extra', 'one_hot', 'fractional', 'nan'])
def test_rejects_ambiguous_fate_batches(kind):
    batch = [torch.zeros(2, 2), torch.zeros(2, 2, 3), torch.ones(2, 2, 1)]
    if kind != 'missing':
        batch.append(torch.ones(2, 2, 2 if kind == 'one_hot' else 1))
    if kind == 'extra':
        batch.append(torch.ones(2, 2, 1))
    if kind == 'fractional':
        batch[3] *= .5
    if kind == 'nan':
        batch[3] *= float('nan')
    with pytest.raises(ValueError, match='[Ff]ate'):
        BatchProcessor(batch, 0, fate_bias=True)


@pytest.mark.parametrize('keys', [['other'], ['other', 'growth'], ['growth', 'other']])
def test_fate_ids_remain_aligned_with_cells(observations, tmp_path, keys):
    model = make_model(observations, keys, 'growth', tmp_path)
    batch = next(iter(DataLoader(model.LitDataModule.train_dataset, batch_size=8)))
    processed = BatchProcessor(batch, 0, fate_bias=True)
    ids = processed.F_idx.astype(int)
    np.testing.assert_allclose(processed.X0.numpy(), observations.X[ids])
    torch.testing.assert_close(
        processed.W_hat[1],
        torch.softmax(torch.tensor(observations.obs['growth'].to_numpy()[ids],
                                   dtype=torch.float32)[:, None] * .1, dim=0),
    )


def test_fate_aware_model_step_preserves_legacy_four_tensor_batch(tmp_path):
    """Exercise the model consumer, so dropping its explicit opt-in fails."""
    from scdiffeq.core.lightning_models import LightningSDE_FateBiasAware

    fate_path = tmp_path / 'fates.csv'
    pd.DataFrame({'A': [1., 0.], 'B': [0., 1.]},
                 index=['12', '21']).to_csv(fate_path)

    # The external annotation graph is replaced with fixed predictions; the
    # model's fate-table lookup, batch processing, integration and loss are real.
    def graph(states, annot_key, query_t):
        return [pd.DataFrame({'A': [1., 0.], 'B': [0., 1.]})]

    model = LightningSDE_FateBiasAware(
        latent_dim=3, mu_hidden=[4], sigma_hidden=[4],
        mu_dropout=0., sigma_dropout=0., backend='tensorized',
        t0_idx=pd.Index(['12', '21']), kNN_Graph=graph,
        fate_bias_csv_path=str(fate_path),
    )
    batch = [torch.tensor([[0., .1], [0., .1]]),
             torch.randn(2, 2, 3), torch.ones(2, 2, 1),
             torch.tensor([[[12.], [13.]], [[21.], [22.]]])]
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert model.X_hat.shape == (2, 2, 3)
