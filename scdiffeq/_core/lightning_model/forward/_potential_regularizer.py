

import torch


from ...utils import AutoParseBase


class PotentialRegularizer(AutoParseBase):
    """
    Requires time_key to use... for now, limited to real time...

    Sources:
    --------
    (1) https://github.com/gifford-lab/prescient/blob/64d4318e5c844eb3a99e290594cb65c9b5900e5e/prescient/train/util.py#L35-L48
    (2) https://github.com/gifford-lab/prescient/blob/64d4318e5c844eb3a99e290594cb65c9b5900e5e/prescient/train/run.py#L136-L148
    """
    
    def __config__(self, kwargs, ignore):
        
        self.__parse__(kwargs, ignore=ignore)
        
        self.time_key = kwargs["model"].hparams['time_key']
        self.func_type = kwargs["model"].hparams['func_type']
        self.adata = kwargs["model"].adata
        self.dt = kwargs["model"].hparams['dt']
        self.use_key = kwargs["model"].hparams['use_key']
        self.df = self.adata.obs.copy()
        self.device = kwargs["model"].device     

    def __init__(self, batch, model):

        self.__config__(locals(), ignore=["self", "model"])
        self.model = model

    @property
    def t_final(self):
        return self.batch.t.max().item()

    @property
    def X_final(self):
        # TODO: expand functionality to fetch data more efficiently from adata
        # due to "use_key" only referncing matrices from adata.obsm, one would not be
        # able to use GEX or other things stored in adata.layers, etc.
        
        bool_idx = self.df[self.time_key] == self.t_final
        
        if self.use_key in self.adata.layers:
            x_ = self.adata[bool_idx].layers[self.use_key]
        elif self.use_key in self.adata.obsm_keys():
            x_ = self.adata[bool_idx].obsm[self.use_key]
            
        return torch.Tensor(x_).to(self.device)

    @property
    def n_cells_t_final(self):
        return self.df.loc[self.df[self.time_key] == self.t_final].shape[0]

    @property
    def batch_size(self):
        return self.batch.X.shape[1]

    @property
    def burn_tspan(self):
        return self.dt * self.burn_steps

    @property
    def burn_t0(self):
        return self.t_final

    @property
    def burn_tf(self):
        return self.burn_t0 + self.burn_tspan

    @property
    def burn_t(self):
        _burn_t = torch.linspace(
            self.burn_t0, self.burn_tf, int(self.burn_steps + 1)
        )  # all steps at resolution dt

        return _burn_t[[0, -1]]

    @property
    def sample_size_factor(self):
        return self.n_cells_t_final / self.batch_size

    def forward_burn(self, _forward, stdev):
        
        """
        Notes:
        ------
        (1) Returns only the final timepoint of the prediction.
        """

        X_burn = _forward(
            X0=self.batch.X0,
            t=self.burn_t,
            dt=self.dt,
            stdev=stdev,
            device=self.device,
        )

        return X_burn[-1]

    def ref_potential(self, func):
        if self.func_type == "TorchNet":
            return func(self.X_final).sum() * -1
        return func.potential(func.mu, self.X_final).sum() * -1

    def burn_potential(self, func, X_burn):
        if self.func_type == "TorchNet":
            return func(X_burn.detach()).sum() * self.sample_size_factor # .detach()
        return func.potential(func.mu, X_burn).sum() * self.sample_size_factor

    def __call__(self, func, ForwardIntegrator, stdev, tau=1, burn_steps=100):
        
        self.__parse__(locals())
        
        X_burn = self.forward_burn(ForwardIntegrator, stdev)
        ref_psi = self.ref_potential(func)
        burn_psi = self.burn_potential(func, X_burn)
        
        """
        fv_tot = pos_fv + neg_fv
        fv_tot *= config.train_tau
        """
        
        return ref_psi, burn_psi
    
    def diff(self, func, ForwardIntegrator, stdev, tau=1, burn_steps=100):
        
        ref_psi, burn_psi = self.__call__(func, ForwardIntegrator, stdev, tau, burn_steps)
        
#         print("\tref: {:.1f}\tburn: {:.1f}\tdiff: {:.1f}\t".format(
#             ref_psi.item(),
#             burn_psi.item(),
#             (ref_psi + burn_psi).item(),
#         ))
        
        self.model.log("ref_psi", ref_psi.item())
        self.model.log("burn_psi", burn_psi.item())
        
        return torch.mul(ref_psi + burn_psi, self.tau) # .abs()
        