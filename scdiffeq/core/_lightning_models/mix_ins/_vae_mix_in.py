
import torch_nets


from ._base_mix_in import BaseMixIn


# -- module class: ---------------------------------------------------------------
class VAEMixIn(BaseMixIn):
    """Gives the basic VAE forward functionality of encode -> sdeint -> decode"""

    def __init__(self, n_features, n_hidden, n_latent=20, *args, **kwargs):
        super(VAEMixIn, self).__init__(*args, **kwargs)

        self.n_latent = n_latent
        self.n_features = n_features
        self.n_hidden = n_hidden

        self.Encoder = torch_nets.Encoder(
            in_features=self.n_features, out_features=self.n_latent, n_hidden=self.n_hidden
        )

        self.Decoder = torch_nets.Decoder(
            in_features=self.n_latent, out_features=self.n_features, n_hidden=self.n_hidden
        )

    def encode(self, X):
        return self.Encoder(X)

    def decode(self, Z):
        return self.Decoder(Z)

    def forward(self, X0, t, stage=None, **kwargs):
        """Overrides the `forward` method from `BaseLightningSDE`"""
        Z_hat = self.integrate(X0=self.encode(X0), t=t, dt=self.hparams["dt"], **kwargs)
        return self.decode(Z_hat)
