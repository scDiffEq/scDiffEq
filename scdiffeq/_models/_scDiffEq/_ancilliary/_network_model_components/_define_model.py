
__module_name__ = "_define_model.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import licorice_font


# import local dependencies #
# ------------------------- #
from .. import _neural_networks as nn
# from ._VariationalAutoEncoder._LinearVAE import _LinearVAE
# from ._Neural_Differential_Equation import Neural_Differential_Equation


def _group_model_hyper_params(
    X_data,
    diffusion,
    VAE_latent_dim,
    VAE_hidden_layers,
    VAE_power,
    VAE_dropout,
    VAE_activation_function_dict,
    drift_hidden_architecture,
    drift_activation_function,
    diffusion_hidden_architecture,
    diffusion_activation_function,
    drift_dropout,
    diffusion_dropout,
    batch_size,
    brownian_size,
):

    ModelParams = {}

    ModelParams["VAE"] = {
        "X_dim": X_data.shape[-1],
        "latent_dim": VAE_latent_dim,
        "hidden_layers": VAE_hidden_layers,
        "power": VAE_power,
        "dropout": VAE_dropout,
        "activation": VAE_activation_function_dict,
    }

    ModelParams["NDE"] = {
        "diffusion": diffusion,
        "in_dim": X_data.shape[-1],
        "out_dim": X_data.shape[-1],
        "drift_hidden_architecture": drift_hidden_architecture,
        "drift_activation_function": drift_activation_function,
        "diffusion_hidden_architecture": diffusion_hidden_architecture,
        "diffusion_activation_function": diffusion_activation_function,
        "drift_dropout": drift_dropout,
        "diffusion_dropout": diffusion_dropout,
        "batch_size": batch_size,
        "brownian_size": brownian_size,
    }

    return ModelParams


def _print_model(encoder, decoder, NDE):

    if encoder:
        licorice_font.underline("Encoder:", ["BOLD", "BLUE"], n_newline=0)
        print(encoder)

    licorice_font.underline("Neural DiffEq:", ["BOLD", "BLUE"], n_newline=0)
    print(NDE)

    if decoder:
        licorice_font.underline("\nDecoder:", ["BOLD", "BLUE"], n_newline=0)
        print(decoder)


def _instantiate_model(VAE_Params, NDE_Params, device, silent=False):

    """
    VAE_Params
    NDE_Params
    device
    silent

    Returns:
    --------
    VAE

    NDE

    parameters

    """


    if VAE_Params != None:

        VAE = nn.LinearVAE(
            X_dim=VAE_Params["X_dim"],
            latent_dim=VAE_Params["latent_dim"],
            hidden_layers=VAE_Params["hidden_layers"],
            power=VAE_Params["power"],
            dropout=VAE_Params["dropout"],
            activation_function_dict=VAE_Params["activation"],
            device=device,
        )

        encoder, decoder = VAE._encoder, VAE._decoder

        encoder_params = list(encoder.parameters())
        decoder_params = list(decoder.parameters())

        NDE_in_dim = NDE_out_dim = VAE_Params["latent_dim"]

    else:
        encoder_params, decoder_params = [], []
        encoder, decoder = None, None
        NDE_in_dim = NDE_Params["in_dim"]
        NDE_out_dim = NDE_Params["out_dim"]
        VAE = None

    NDE = nn.NeuralDiffEq(
        diffusion=NDE_Params["diffusion"],
        in_dim=NDE_in_dim,
        out_dim=NDE_out_dim,
        drift_hidden_architecture=NDE_Params["drift_hidden_architecture"],
        drift_activation_function=NDE_Params["drift_activation_function"],
        diffusion_hidden_architecture=NDE_Params["diffusion_hidden_architecture"],
        diffusion_activation_function=NDE_Params["diffusion_activation_function"],
        drift_dropout=NDE_Params["drift_dropout"],
        diffusion_dropout=NDE_Params["diffusion_dropout"],
        batch_size=NDE_Params["batch_size"],
        brownian_size=NDE_Params["brownian_size"],
    )

    if not silent:
        _print_model(encoder, decoder, NDE)

    parameters = list(NDE.parameters()) + encoder_params + decoder_params

    return VAE, NDE, parameters


def _define_model(
    X_data,
    device,
    silent,
    use_key,
    use_layer,
    optimization_func,
    VAE,
    VAE_latent_dim,
    VAE_hidden_layers,
    VAE_power,
    VAE_dropout,
    VAE_activation_function_dict,
    drift_hidden_architecture,
    drift_activation_function,
    diffusion_hidden_architecture,
    diffusion,
    diffusion_activation_function,
    drift_dropout,
    diffusion_dropout,
    batch_size,
    brownian_size,
    reconstruction_loss_function,
    reparameterization_loss_function,
):
    ModelHyperParams = _group_model_hyper_params(
        X_data=X_data,
        diffusion=diffusion,
        VAE_latent_dim=VAE_latent_dim,
        VAE_hidden_layers=VAE_hidden_layers,
        VAE_power=VAE_power,
        VAE_dropout=VAE_dropout,
        VAE_activation_function_dict=VAE_activation_function_dict,
        drift_hidden_architecture=drift_hidden_architecture,
        drift_activation_function=drift_activation_function,
        diffusion_hidden_architecture=diffusion_hidden_architecture,
        diffusion_activation_function=diffusion_activation_function,
        drift_dropout=drift_dropout,
        diffusion_dropout=diffusion_dropout,
        batch_size=batch_size,
        brownian_size=brownian_size,
    )

    if not VAE:
        ModelHyperParams["VAE"] = None

    VAE, NeuralDiffEq, parameters = _instantiate_model(
        NDE_Params=ModelHyperParams["NDE"],
        VAE_Params=ModelHyperParams["VAE"],
        device=device,
        silent=silent,
    )
    
    Model = {"NeuralDiffEq":NeuralDiffEq, 
             "VAE": VAE,
             "params": parameters,
             "HyperParams":ModelHyperParams,
             "optim":optimization_func,
             "reconst_loss_func":reconstruction_loss_function,
             "reparam_loss_func":reparameterization_loss_function,
            }

    return Model