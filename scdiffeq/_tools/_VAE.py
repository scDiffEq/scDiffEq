
import adata_sdk
from collections import OrderedDict
from IPython import display
import matplotlib.pyplot as plt
import licorice
import numpy as np
import os
import torch
import umap
import vinplots
import pydk
from .._utilities._save_torch_model import _save_torch_model

def _power_space(start, stop, n, power):
    
    start = np.power(start, 1/float(power))
    stop = np.power(stop, 1/float(power))
    
    return np.power( np.linspace(start, stop, num=n), power)

def _get_sequenced_VAE_layer_n(input_dim, output_dim, layers, power):
    
    return _power_space(start=output_dim, stop=input_dim, n=layers, power=power).astype(int)

def _compose_multilayered_nn_sequential(nodes_by_layer, dropout):
    
    """"""    
    neural_net = OrderedDict()
    
    for i in range(len(nodes_by_layer)-1):
        neural_net["{}_layer".format(i)] = torch.nn.Linear(nodes_by_layer[i], nodes_by_layer[i+1])
        if i != len(nodes_by_layer) -2:
            if dropout:
                neural_net["{}_dropout".format(i)] = torch.nn.Dropout(dropout)
            neural_net["{}_LeakyReLU".format(i)] = torch.nn.LeakyReLU()
            
    return torch.nn.Sequential(neural_net)

def _build_encoder_decoder(input_dim, output_dim, layers, power, dropout):
    
    """"""
    
    encoder_nodes_by_layer = _get_sequenced_VAE_layer_n(input_dim, output_dim, layers, power).astype(int)[::-1]
    decoder_nodes_by_layer = _get_sequenced_VAE_layer_n(input_dim, output_dim, layers, power).astype(int)

    encoder = _compose_multilayered_nn_sequential(encoder_nodes_by_layer, dropout)
    decoder = _compose_multilayered_nn_sequential(decoder_nodes_by_layer, dropout)
    
    return encoder, decoder

def live_loss_plot(loss_vals):
    
    display.clear_output(wait=True)
    
    fig = vinplots.Plot()
    fig.construct(nplots=1, ncols=1)
    fig.modify_spines(ax="all", spines_to_delete=['top', 'right'])
    ax = fig.AxesDict[0][0]
    ax.set_ylim(-0.5, np.round(max(loss_vals))+1)
    ax.plot(loss_vals, color='darkred')
    ax.hlines(y=0, xmin=0, xmax=len(loss_vals), ls="--", color="dimgrey", alpha=0.5)
    plt.show()
    
def _epoch(self, plot, outpath, verbose):
    
    self._optimizer.zero_grad()
    X_latent = self._encoder(self._X.to(self._device))
    self._X_decode = self._decoder(X_latent.to(self._device))
    loss = self._LossFunc(self._X_decode.to(self._device), self._X.to(self._device)).to(self._device)
    self._loss_tracker.append(loss.item())
    loss.backward()
    self._optimizer.step()
    if plot:
        live_loss_plot(self._loss_tracker)
    
    if min(self._loss_tracker) == self._loss_tracker[-1]:
        self._X_latent = X_latent
        self._best_epoch = len(self._loss_tracker)
        _save_VAE(self._encoder, self._decoder, outpath, verbose)
        
    self._epoch_counter += 1
        

def _embed(adata, X_latent):
    
    """"""
    adata.obsm['X_vae'] = X_latent
    umap_transformer = umap.UMAP(n_components=2)
    adata.obsm['X_vae_umap'] = umap_transformer.fit_transform(X_latent.cpu().detach().numpy())
    
    return adata
    
def _build_plot(figsize=1.7):
    
    fig = vinplots.Plot()
    fig.construct(nplots=1, ncols=1, figsize=figsize)
    fig.modify_spines(ax="all", spines_to_delete=['top', 'right', 'bottom', 'left'])
    ax = fig.AxesDict[0][0]
    ax.set_xticks([])
    ax.set_yticks([])
    
    return fig, ax

def _plot_umap_embedding(adata, groupby="Annotation"):
    
    """"""
    
    fig, ax = _build_plot(figsize=1.7)
    grouped_by = adata.obs.groupby(groupby)
    umap = adata.obsm['X_vae_umap']
    cmap = vinplots.palettes.Weinreb2020()
    c = list(cmap.values())
    if groupby == "Annotation":
        iterable = list(cmap.keys())
    else:
        iterable = list(grouped_by.groups.keys())
        
    for n, group in enumerate(iterable):
        group_df = grouped_by.get_group(group)
        idx = group_df.index.astype(int)
        if type(group) is int or  type(group) is float:
            if group == min(iterable):
                zorder = 0
            else:
                zorder = 10
        elif group == 'undiff' or group == min(iterable):
            zorder = 0
        else:
            zorder= 10
        ax.scatter(umap[idx,0], umap[idx,1], label=group, color=c[n], zorder=zorder, s=5)
    
    plt.legend(edgecolor='white', markerscale=2)
    
def _save_VAE(encoder, decoder, outpath, verbose=False):
    
    outdir = os.path.join(outpath, "VAE")
    pydk.mkdir_flex(outdir)
    encoder_outpath = os.path.join(outdir, "best.encoder")
    decoder_outpath = os.path.join(outdir, "best.decoder")
    
    if verbose:
        # not a very helpful message until the end.
        msg = licorice.font_format("Saving to", ['BOLD'])
        print("{}: {}".format(msg, outdir))
        print("\t   {}".format(encoder_outpath))
        print("\t   {}".format(decoder_outpath))
    
    _save_torch_model(encoder, encoder_outpath)
    _save_torch_model(decoder, decoder_outpath)
    
    return encoder_outpath, decoder_outpath

class _VAE:
    def __init__(self, adata, n_latent_dims=20, n_layers=5, power=2, dropout=0.1, device=0, lr=1e-3):
        
        self._lr = lr
        self._epoch_counter = 0
        self._X = torch.Tensor(adata.X.toarray())
        self._adata = adata
        self._n_features = self._adata.shape[1]
        self._n_latent_dims = n_latent_dims
        self._n_layers = n_layers
        self._device = device
        self._power = power
        self._dropout = dropout
        self._encoder, self._decoder = _build_encoder_decoder(self._n_features, self._n_latent_dims, self._n_layers, power, dropout)
        self._encoder.to(self._device)
        self._decoder.to(self._device)
        self._LossFunc = torch.nn.MSELoss()
        self._model_params = list(self._encoder.parameters()) + list(self._decoder.parameters())
        self._optimizer = torch.optim.Adam(self._model_params, lr=self._lr)
        self._loss_tracker = []
        
    def learn(self, n_epochs=200, lr=False, plot=True, outpath="./", verbose=False):
                
        self._outpath = outpath
        
        if lr:
            self._lr = lr
            self._optimizer = torch.optim.Adam(self._model_params, lr=self._lr)

        for epoch in range(n_epochs):
            _epoch(self, plot, outpath, verbose)
                
    def embed(self, inplace=True):

        self._adata = _embed(self._adata, self._X_latent)

        if inplace:
            adata = self._adata
        else:
            return self._adata

    def plot_umap(self, groupby="Annotation"):

        _plot_umap_embedding(self._adata, groupby)
        
        
    def save(self, verbose=True):
        
        _save_VAE(encoder, decoder, self._outpath, verbose=False)
        
        adata.obsm['X_vae'] = adata.obsm['X_vae'].cpu().detach().numpy()
        adata_sdk.write_loaded_h5ad(adata, os.path.join(self._outpath, "vae_adata"))