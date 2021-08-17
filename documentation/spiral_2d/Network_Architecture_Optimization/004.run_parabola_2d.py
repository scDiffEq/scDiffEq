
#### PARSE ARGUMENTS ####
import argparse

parser = argparse.ArgumentParser(description="run script info.")
parser.add_argument(
    "--layers",
    dest="layers",
    type=str,
    help="Number of layers in the DiffEq model neural network.",
)
parser.add_argument(
    "--nodes",
    dest="nodes",
    type=str,
    help="Nustrmber of nodes in each fully connected layer in the DiffEq model neural network.",
)

args = parser.parse_args()

N_LAYERS=int(args.layers)
N_NODES=int(args.nodes)

print("Number of layers: " + str(N_LAYERS))
print("Number of nodes / layer: " + str(N_NODES))
#########################

#### IMPORT / SETUP ####
import scdiffeq as sdq

odeint, torch, np, pd, plt, nn, a, os, time, optim, sp, PCA, v = sdq.ut.devlibs()

OUTPATH = "/home/mvinyard/results/scdiffeq/parabola_2d/neural_ODEs/run004_network_architecture_analysis/" + "004." + str(N_LAYERS) + "_" + str(N_NODES) + "/"
N_TRAJ=250
N_SAMPLES=40
TIME_SPAN=1
N_EPOCHS=2500
VIZ_FREQ=50
VAL_FREQ=50
########################

#### SIMULATE SOME DATA ####
Simulation = sdq.data.GenericSimulator(save_dir=OUTPATH)
Simulation.set_initial_conditions_sampling_distribution("normal")
Simulation.get_initial_conditions(
    loc=0,
    scale=2,
    size=[N_TRAJ, 2],
    plot=True,
)  # i.e., normal(loc=0.0, scale=1.0, size=None)
Simulation.create_time_vector(time_span=TIME_SPAN, n_samples=N_SAMPLES)
Simulation.simulate_ODE("parabola_2d")
############################

#### SETUP DIFFEQ ####
DiffEq = sdq.tl.scDiffEq(n_layers=N_LAYERS, outdir=OUTPATH, nodes_n=N_NODES, nodes_m=N_NODES)
DiffEq.preflight(Simulation.adata)
DiffEq.evaluate()
#######################

#### LEARN DIFFEQ ####
DiffEq.learn(n_epochs=N_EPOCHS,
             validation_frequency=VAL_FREQ,
             visualization_frequency=VIZ_FREQ,
             notebook=False)
#######################

#### EVALUATE DIFFEQ ####
DiffEq.evaluate(reset_KernelDensity=True)
#########################

#### SAVE ####
DiffEq.save(outdir=OUTPATH)
##############

#### COMPUTE QUASI-POTENTIAL LANDSCAPE ####
DiffEq.compute_quasi_potential()
###########################################
