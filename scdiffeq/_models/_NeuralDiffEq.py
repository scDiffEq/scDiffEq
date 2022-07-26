
import torch

class NeuralDiffEq(torch.nn.Module):

    noise_type = "general"
    sde_type = "ito"

    def __init__(self, mu, sigma=False, brownian_size=1):
        super(NeuralDiffEq, self).__init__()

        self.mu = mu
        self.sigma = sigma
        self._brownian_size = brownian_size

    def f(self, t, y0):
        return self.mu(y0)

    def forward(self, t, y0):
        return self.mu(y0)

    def g(self, t, y0):
        return _sigma(self.sigma, y0, self._brownian_size)
