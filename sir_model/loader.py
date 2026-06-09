import json
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def LOAD_parameters(namefile = "./parameters_test.json"):
    with open("./parameters_test.json") as f:
        temp = json.load(f)
    mus, taus, xis, PARAMS = temp["mus"], temp["taus"], temp["xis"], temp["PARAMS"]
    rect_coords_M = [mus["pol"][0], mus["h"][0],
                    mus["pol"][2] - mus["pol"][0], mus["h"][2] - mus["h"][0]]
    rect_coords_T = [taus["pol"][0], taus["h"][0],
                    taus["pol"][2] - taus["pol"][0], taus["h"][2] - taus["h"][0]]
    rect_coords_V = [xis["pol"][0], xis["h"][0],
                    xis["pol"][2] - xis["pol"][0], xis["h"][2] - xis["h"][0]]

    simulated_days = 3000
    
    return mus, taus, xis, PARAMS, rect_coords_M, rect_coords_T, rect_coords_V, simulated_days

def LOAD_plots_specification():
    hot_r = plt.get_cmap("hot_r")
    my_map = ListedColormap(hot_r(np.linspace(0, 1, 51)))
    my_map.set_bad(color='gray')

    contour_colors = ['#000', '#000', '#000']
    Lx, Ly = 1.83, 1.73

    return my_map, Lx, Ly, contour_colors


def LOAD_sweep_config(PARAMS):
    NP = 100   # polarization grid
    NB = 100   # homophily grid
    PS = 5     # n_groups (population size in old code)
    betas = [0.15, 0.25, 0.45]

    # Endpoints nudged slightly off 0 and 1 to avoid degenerate Beta params
    pol_vals = jnp.linspace(0.01, 0.99, NP)
    h_vals   = jnp.linspace(0.0, 6.0, NB)


    def build_state0_3(pop):
        init_inf = 1e-6
        return jnp.stack([pop * (1 - init_inf), pop * init_inf, jnp.zeros_like(pop)])

    def build_state0_4(pop):
        init_inf = 1e-6
        Z = jnp.zeros_like(pop)
        return jnp.stack([pop * (1 - init_inf), pop * init_inf, Z, Z])


    beh_M = jnp.linspace(0, PARAMS.get('mu_max', 1.0), PS)            # masks: 0..1
    beh_T = jnp.linspace(0, PARAMS.get('testing_rate_max', 0.05), PS) # testing
    beh_V = jnp.linspace(0, PARAMS.get('vaccination_rate_max', 0.05), PS)  # vacc

    return NP, NB, PS, betas, pol_vals, h_vals, build_state0_3, build_state0_4, beh_M, beh_T, beh_V