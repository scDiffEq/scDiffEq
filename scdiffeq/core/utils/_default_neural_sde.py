

from neural_diffeqs import NeuralSDE


def default_NeuralSDE(
    state_size,
    mu_hidden=[1600, 1600],
    sigma_hidden=[800, 800],
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

):
    return NeuralSDE(
        state_size=state_size,
        mu_hidden=mu_hidden,
        sigma_hidden=sigma_hidden,
        mu_activation=mu_activation,
        sigma_activation=sigma_activation,
        mu_dropout=mu_dropout,
        sigma_dropout=sigma_dropout,
        mu_bias=mu_bias,
        sigma_bias=sigma_bias,
        mu_output_bias=mu_output_bias,
        sigma_output_bias=sigma_output_bias,
        mu_potential=mu_potential,
        sigma_potential=sigma_potential,
        noise_type=noise_type,
        sde_type=sde_type,
        brownian_size=brownian_size,
    )