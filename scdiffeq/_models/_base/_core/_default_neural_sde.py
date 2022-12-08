
from neural_diffeqs import NeuralSDE

def default_NeuralSDE(state_size):
    return NeuralSDE(
        state_size=state_size,
        mu_hidden=[400, 400],
        sigma_hidden=[400, 400],
        mu_activation="LeakyReLU",
        sigma_activation="LeakyReLU",
        mu_dropout=0.2,
        sigma_dropout=0.2,
        mu_bias=True,
        sigma_bias=True,
        mu_output_bias=True,
        sigma_output_bias=True,
        mu_potential=False,
        sigma_potential=False,
        noise_type="general",
        sde_type="ito",
        brownian_size=1,
    )