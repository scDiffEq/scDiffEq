import os

n_nodes_range = [1, 2, 3, 4, 5, 10, 20, 25, 50, 100]
n_layer_range = range(1,10)

for layer_n in n_layer_range:
    for nodes_n in n_nodes_range:
        print(layer_n, nodes_n)
        command = "python /home/mvinyard/scripts/scdiffeq_runscripts/parabola_2d/004.run_parabola_2d.py --layers {} --nodes {} &".format(layer_n, nodes_n)
        os.system(command)
