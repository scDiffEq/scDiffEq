from collections import OrderedDict
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
import vinplots

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
    
def _epoch(self):
    
    self._optimizer.zero_grad()
    X_latent = self._encoder(self._X.to(self._device))
    self._X_decode = self._decoder(X_latent.to(self._device))
    loss = self._LossFunc(self._X_decode.to(self._device), self._X.to(self._device)).to(self._device)
    self._loss_tracker.append(loss.item())
    loss.backward()
    self._optimizer.step()
    live_loss_plot(self._loss_tracker)
    
    if min(self._loss_tracker) == self._loss_tracker[-1]:
        self._X_latent = X_latent
        self._best_epoch = len(self._loss_tracker)
        

def _embed(adata, X_latent):
    
    """"""
    adata.obsm['X_vae'] = X_latent
    umap_transformer = umap.UMAP(n_components=2)
    adata.obsm['X_vae_umap'] = umap_transformer.fit_transform(X_latent.cpu().detach().numpy())
    
    return adata
    
def _build_plot(figsize=1.5):
    
    fig = vinplots.Plot()
    fig.construct(nplots=1, ncols=1, figsize=figsize)
    fig.modify_spines(ax="all", spines_to_delete=['top', 'right', 'bottom', 'left'])
    ax = fig.AxesDict[0][0]
    ax.set_xticks([])
    ax.set_yticks([])
    
    return fig, ax

def _plot_umap_embedding(adata):
    
    """"""
    c = vinplots.palettes.Weinreb2020()
    fig, ax = _build_plot(figsize=1.5)
    grouped_by = adata.obs.groupby('Annotation')
    umap = adata.obsm['X_vae_umap']

    for annot in c.keys():
        try:
            annot_df = grouped_by.get_group(annot)
            idx = annot_df.index.astype(int)
            if annot == 'undiff':
                zorder = 0
            else:
                zorder= 10
            ax.scatter(umap[idx,0], umap[idx,1], label=annot, color=c[annot], zorder=zorder, s=5)
        except:
            continue
    plt.legend(edgecolor='white', markerscale=2)
    
class _VAE:
    def __init__(self, adata, n_latent_dims=20, n_layers=5, power=2, dropout=0.1, device=0):
        
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
        self._optimizer = torch.optim.Adam(self._model_params, lr=0.001)
        self._loss_tracker = []
        
    def learn(self, n_epochs=200):
        
        """"""

        for epoch in range(n_epochs):
            _epoch(self)
        
    def embed(self, inplace=True, plot=True):
        self._adata = _embed(self._adata, self._X_latent)
        if plot:
            _plot_umap_embedding(self._adata)
        
        if inplace:
            adata = self._adata
        else:
            return self._adata
        