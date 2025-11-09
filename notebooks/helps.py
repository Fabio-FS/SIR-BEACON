from src.models import SIRM, SIRT, SIRV
from src.utils.batch_sweep import sweep_two_parameters
from src.utils.visualization.trajectory import run_single_simulation

from src.utils.visualization import *
from src.utils.visualization.core import Lx, Ly
import numpy as np
import matplotlib.pyplot as plt

import json
import ast


CH = ["#00441b", "#238b45", "#000", "#66c2a4", "#99d8c9"]       # used for fixed homophily
CP = ["#000", "#7f0000", "#d7301f", "#fc8d59", "#fdbb84"]       # used for fixed polarization


colors_X = ['#66c2a4', '#238b45','#00441b']  # fixed polarization
colors_Y = ['#67001f', '#e7298a', '#df65b0']  # fixed homophily


def read_json(file_path):
    """
    Read model parameters from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing model parameters
        
    Returns:
        Dictionary of model parameters
    """
    with open(file_path, 'r') as file:
        params = json.load(file)
    
    # Convert tuple strings to actual tuples if needed
    for key, value in params.items():
        if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
            params[key] = ast.literal_eval(value)
    
    return params



def find_closest_pol_index(results, target_pol_value):
    """
    Find the index where polarization is closest to target_pol_value
    
    Args:
        results: Dictionary returned by sweep_two_parameters
        target_pol_value: The polarization value you're looking for
        
    Returns:
        Index in the flattened array where polarization is closest to target
    """
    # Get polarization values from the parameter grid
    param1_name = results['parameter_names']['param1']
    
    if param1_name == "beta_params":
        # Polarization values are in param1 (first dimension)
        pol_values = np.linspace(
            results['parameter_ranges']['param1']['m'],
            results['parameter_ranges']['param1']['M'],
            results['parameter_ranges']['param1']['n']
        )
        # Find closest index
        closest_idx = np.argmin(np.abs(pol_values - target_pol_value))
        return closest_idx
    else:
        # Polarization values are in param2 (second dimension)
        pol_values = np.linspace(
            results['parameter_ranges']['param2']['m'],
            results['parameter_ranges']['param2']['M'],
            results['parameter_ranges']['param2']['n']
        )
        # Find closest index
        closest_idx = np.argmin(np.abs(pol_values - target_pol_value))
        return closest_idx
    

def find_closest_hom_index(results, target_hom_value):
    """
    Find the index where homophilic_tendency is closest to target_hom_value
    
    Args:
        results: Dictionary returned by sweep_two_parameters
        target_hom_value: The homophilic_tendency value you're looking for
        
    Returns:
        Index in the flattened array where homophilic_tendency is closest to target
    """
    # Get parameter names to determine which one is homophilic_tendency
    param1_name = results['parameter_names']['param1']
    param2_name = results['parameter_names']['param2']
    
    if param1_name == "homophilic_tendency":
        # Homophilic tendency values are in param1 (first dimension)
        hom_values = np.linspace(
            results['parameter_ranges']['param1']['m'],
            results['parameter_ranges']['param1']['M'],
            results['parameter_ranges']['param1']['n']
        )
        # Find closest index
        closest_idx = np.argmin(np.abs(hom_values - target_hom_value))
        return closest_idx
    elif param2_name == "homophilic_tendency":
        # Homophilic tendency values are in param2 (second dimension)
        hom_values = np.linspace(
            results['parameter_ranges']['param2']['m'],
            results['parameter_ranges']['param2']['M'],
            results['parameter_ranges']['param2']['n']
        )
        # Find closest index
        closest_idx = np.argmin(np.abs(hom_values - target_hom_value))
        return closest_idx
    else:
        # Homophilic tendency not found in parameter names
        raise ValueError("homophilic_tendency not found in parameter names")
    
def plot_fixed_homophily(results, H, model, PARAM, name):
    PARAM["fixed_mean"] = 0.5
    Indx_H0 = find_closest_hom_index(results, H[0])
    Indx_H1 = find_closest_hom_index(results, H[1])
    Indx_H2 = find_closest_hom_index(results, H[2])

    R_0 = np.sum(results['final_state']['R'][Indx_H0,:], axis=1)
    R_1 = np.sum(results['final_state']['R'][Indx_H1,:], axis=1)
    R_2 = np.sum(results['final_state']['R'][Indx_H2,:], axis=1)


    P_almost_consensus = 0.000001
    baseline  = sweep_two_parameters(
            model_module=model,
            param1_name='beta_params',
            param1_range=[P_almost_consensus], 
            param2_name='homophilic_tendency',
            param2_range=[0],
            custom_base_params=PARAM,
            population_size = 5,
        )
    BS = np.sum(baseline['final_state']['R'][Indx_H0,:], axis=1)
    pol_values = np.linspace(0, 1, len(R_0))

    LW = 2

    fig, ax = plt.subplots(1, 1, figsize=(Lx, Ly))

    ax.plot([0,1], [BS, BS], color='black', linestyle='--')
    ax.plot(pol_values, R_0, color = colors_Y[0], linewidth=LW)
    ax.plot(pol_values, R_1, color = colors_Y[1], linewidth=LW)
    ax.plot(pol_values, R_2, color = colors_Y[2], linewidth=LW)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])

    ax. set_xticklabels([])
    ax. set_yticklabels([])

    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # fontsize of ticks
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.patch.set_visible(False)  # Make figure background transparent

    plt.savefig(name)

def plot_fixed_polarization(results, P, model, PARAM, name):
    PARAM["fixed_mean"] = 0.5
    Indx_P0 = find_closest_pol_index(results, P[0])
    Indx_P1 = find_closest_pol_index(results, P[1])
    Indx_P2 = find_closest_pol_index(results, P[2])

    R_0 = np.sum(results['final_state']['R'][:,Indx_P0], axis=1)
    R_1 = np.sum(results['final_state']['R'][:,Indx_P1], axis=1)
    R_2 = np.sum(results['final_state']['R'][:,Indx_P2], axis=1)


    P_almost_consensus = 0.000001
    baseline  = sweep_two_parameters(
            model_module=model,
            param1_name='beta_params',
            param1_range=[P_almost_consensus], 
            param2_name='homophilic_tendency',
            param2_range=[0],
            custom_base_params=PARAM,
            population_size = 5,
        )
    BS = np.sum(baseline['final_state']['R'][:,Indx_P2], axis=1)
    print("BS = ", BS)
    hom_values = np.linspace(0, 5, len(R_0))

    LW = 2

    fig, ax = plt.subplots(1, 1, figsize=(Lx, Ly))

    ax.plot([0,6], [BS, BS], color='black', linestyle='--')
    ax.plot(hom_values, R_0, color = colors_X[0], linewidth=LW)
    ax.plot(hom_values, R_1, color = colors_X[1], linewidth=LW)
    ax.plot(hom_values, R_2, color = colors_X[2], linewidth=LW)

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)

    ax.set_xticks([0, 3, 6])
    ax.set_yticks([0, 0.5, 1])

    ax. set_xticklabels([])
    ax. set_yticklabels([])

    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # fontsize of ticks
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.patch.set_visible(False)  # Make figure background transparent

    plt.savefig(name)



def find_hpol_minmax(model, ranges, PARAMS):
    NB_quick = 40
    pol_range = {"m": ranges["pol"][0], "M": ranges["pol"][2], "n": NB_quick}
    homophilic_tendency = {"m": ranges["h"][0], "M": ranges["h"][2], "n": NB_quick}
    PARAMS["fixed_mean"] = ranges["mean"][1]
    RM_1 = sweep_two_parameters(
        model_module=model,
        param1_name="beta_params",           # parameter 1 name
        param1_range=pol_range,    # parameter 1 range
        param2_name="homophilic_tendency",      # parameter 2 name
        param2_range=homophilic_tendency,         # parameter 2 range
        custom_base_params=PARAMS,
        simulated_days=1000,
        population_size=5,
        batch_size=1000
    )

    # find the minimum and maximum of the metrics
    min_value = np.min(np.sum(RM_1['final_state']['R']+RM_1['final_state']['I'], axis=2))
    max_value = np.max(np.sum(RM_1['final_state']['R']+RM_1['final_state']['I'], axis=2))

    R_data = np.array(np.sum(RM_1['final_state']['R']+RM_1['final_state']['I'], axis=2))
    min_idx = np.unravel_index(np.argmin(R_data), R_data.shape)
    max_idx = np.unravel_index(np.argmax(R_data), R_data.shape)
    param1_grid = RM_1['parameter_grid']['param1_vals']
    param2_grid = RM_1['parameter_grid']['param2_vals']
    min_pol = param1_grid[min_idx]
    min_hom = param2_grid[min_idx]
    max_pol = param1_grid[max_idx]
    max_hom = param2_grid[max_idx]

    print(f"Minimum R+I value: {min_value:.4f}")
    print(f"   at polarization = {min_pol:.4f}, homophily = {min_hom:.4f}")
    print(f"Maximum R+I value: {max_value:.4f}")
    print(f"   at polarization = {max_pol:.4f}, homophily = {max_hom:.4f}")

    return [min_pol, min_hom], [max_pol, max_hom]

def calc_minmax_trajectories(model, min_hom_pol, max_hom_pol, mean, PARAMS, simulated_days=1000):

    _ , I_min, R_min, *_ = run_single_simulation(min_hom_pol[0], min_hom_pol[1], mean, PARAMS, model, simulated_days=simulated_days)
    _ , I_max, R_max, *_ = run_single_simulation(max_hom_pol[0], max_hom_pol[1], mean, PARAMS, model, simulated_days=simulated_days)
    _ , I_base, R_base, *_ = run_single_simulation(0.0001, 0, mean, PARAMS, model, simulated_days=simulated_days)
    _ , I_OG, R_OG, *_ =       run_single_simulation(0.0001, 0, 0.5, PARAMS, model, simulated_days=simulated_days)

    return [I_min, R_min], [I_max, R_max], [I_base, R_base], [I_OG, R_OG]


def plot_double_comparison(days, mins, maxs, bases, OG, pathname, Lx, Ly):
    fig, ax = plt.subplots(1, 1, figsize=(Lx, Ly))

    Im, Rm = mins[0], mins[1]
    IM, RM = maxs[0], maxs[1]

    Ib, Rb = bases[0], bases[1]     # no info about distribution of behavior only average, beta and gamma and range
    IG, RG = OG[0], OG[1]           # no info about average behavior, only about beta and gamma and range
    
    ax.fill_between(days, Rm+Im, RM+IM, color='black', alpha=0.5)
    ax.plot(days, Rm+Im, color ="black", linewidth=0.5)
    ax.plot(days, RM+IM, color ="black", linewidth=0.5)
    ax.plot(days, Rb+Ib, '--',color ="black")
    ax.plot(days, RG + IG, ':', color ="black")

    ax.set_xlim(0, 1000)
    ax.set_ylim(-0.01, 0.6)

    ax.set_xticks([0,  500, 1000])
    ax.set_yticks([0, 0.3, 0.6])

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # make background transparent
    fig.patch.set_visible(False)  # Make figure background transparent

    # save figure
    plt.savefig(pathname)
    return fig



def plot_comparison(days, Im, Rm, IM, RM, Ib, Rb, pathname, Lx, Ly):
    fig, ax = plt.subplots(1, 1, figsize=(Lx, Ly))

    ax.fill_between(days, Rm+Im, RM+IM, color='black', alpha=0.5)
    ax.plot(days, Rm+Im, color ="black", linewidth=0.5)
    ax.plot(days, RM+IM, color ="black", linewidth=0.5)
    ax.plot(days, Rb+Ib, '--',color ="black")

    ax.set_xlim(0, 1000)
    ax.set_ylim(-0.01, 0.6)

    ax.set_xticks([0,  500, 1000])
    ax.set_yticks([0, 0.3, 0.6])

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # make background transparent
    fig.patch.set_visible(False)  # Make figure background transparent

    # save figure
    plt.savefig(pathname)
    return fig

def calc_minmax_trajectories2(model, mins, maxs, mean, PARAMS, simulated_days = 1000):
    
    population_size = 5
    initial_infected_prop = 0.001
    use_contact_matrix = True  # Set to True to use homophily
    return_trajectory = True   # Important: set this to True to get the full trajectory
    # mins:

    params = PARAMS.copy()
    params.update({
        'homophilic_tendency': mins[1],   # Homophily parameter
        'fixed_mean': mean             # Mean of the distribution
    })
    polarization = maxs[0]  # Polarization parameter

    from src.utils.distributions import pol_mean_to_ab
    import jax.numpy as jnp
    beta_params = pol_mean_to_ab(polarization, params['fixed_mean'])
    

    states_trajectory, r0, homophily = SIRM.run_simulation(
        beta_params=beta_params,
        params=params,
        simulated_days=simulated_days,
        initial_infected_prop=initial_infected_prop,
        population_size=population_size,
        use_contact_matrix=use_contact_matrix,
        return_trajectory=return_trajectory
    )
    S_min, I_min, R_min = states_trajectory

    # You can calculate the total values by summing across population compartments
    Sm = jnp.sum(S_min, axis=1)
    Im = jnp.sum(I_min, axis=1)
    Rm = jnp.sum(R_min, axis=1)



    # maxs:

    params = PARAMS.copy()
    params.update({
        'homophilic_tendency': maxs[1],   # Homophily parameter
        'fixed_mean': mean             # Mean of the distribution
    })
    polarization = maxs[0]  # Polarization parameter

    beta_params = pol_mean_to_ab(polarization, params['fixed_mean'])

    states_trajectory, r0, homophily = SIRM.run_simulation(
        beta_params=beta_params,
        params=params,
        simulated_days=simulated_days,
        initial_infected_prop=initial_infected_prop,
        population_size=population_size,
        use_contact_matrix=use_contact_matrix,
        return_trajectory=return_trajectory
    )
    S_max, I_max, R_max = states_trajectory

    # You can calculate the total values by summing across population compartments
    SM = jnp.sum(S_max, axis=1)
    IM = jnp.sum(I_max, axis=1)
    RM = jnp.sum(R_max, axis=1)


    # baseline: pol = 0.00001, homophily = 0, mean = 0.5
    params = PARAMS.copy()
    params.update({
        'homophilic_tendency': 0,   # Homophily parameter
        'fixed_mean': 0.5             # Mean of the distribution
    })
    polarization = 0.000001  # Polarization parameter

    beta_params = pol_mean_to_ab(polarization, params['fixed_mean'])

    states_trajectory, r0, homophily = SIRM.run_simulation(
        beta_params=beta_params,
        params=params,
        simulated_days=simulated_days,
        initial_infected_prop=initial_infected_prop,
        population_size=population_size,
        use_contact_matrix=use_contact_matrix,
        return_trajectory=return_trajectory
    )
    S_baseline, I_baseline, R_baseline = states_trajectory

    Sb = jnp.sum(S_baseline, axis=1)
    Ib = jnp.sum(I_baseline, axis=1)
    Rb = jnp.sum(R_baseline, axis=1)


    return Sm, Im, Rm, SM, IM, RM, Sb, Ib, Rb
def plot_min_max_trajectories(model, params, p_min, p_max, sim_days=1000, pop_size=5):

    
    # Create copies of parameters
    min_params = dict(params)
    max_params = dict(params)
    
    # Set homophilic tendency for min and max
    min_params['homophilic_tendency'] = p_min[1]
    max_params['homophilic_tendency'] = p_max[1]
    
    # Convert polarization to beta parameters
    min_alpha = pol_to_alpha(p_min[0])
    max_alpha = pol_to_alpha(p_max[0])
    
    # Create beta params tuples (symmetric distributions)
    min_beta_params = (min_alpha, min_alpha)
    max_beta_params = (max_alpha, max_alpha)
    
    # Run simulations with trajectory output
    min_results = model.run_simulation(
        beta_params=min_beta_params,
        params=min_params,
        simulated_days=sim_days,
        initial_infected_prop=1e-4,
        population_size=pop_size,
        use_contact_matrix=True,
        return_trajectory=True,
    )
    
    max_results = model.run_simulation(
        beta_params=max_beta_params,
        params=max_params,
        simulated_days=sim_days,
        initial_infected_prop=1e-4,
        population_size=pop_size,
        use_contact_matrix=True,
        return_trajectory=True,
    )
    
    # Extract trajectories (first element of results)
    min_trajectory = min_results[0]
    max_trajectory = max_results[0]
    
    # Get the I compartment (index 1 - second compartment)
    i_compartment_idx = 1
    
    # Create time steps
    dT = params.get('dT', 1)
    time_points = np.arange(0, sim_days + dT, dT)
    
    # Plot using the provided plotting function
    trajectory_dict = {
        f"Min (Pol={p_min[0]:.2f}, H={p_min[1]:.2f})": min_trajectory[i_compartment_idx],
        f"Max (Pol={p_max[0]:.2f}, H={p_max[1]:.2f})": max_trajectory[i_compartment_idx]
    }
    
    # Plot the trajectories
    fig = plot_time_series(
        trajectory=trajectory_dict,
        time_points=time_points[:len(min_trajectory[i_compartment_idx])],
        title=f"{model.get_compartment_info()[0]} Model: I(t) for Min/Max Infection Parameters",
        colors=["blue", "red"],
        grid=False
    )
    
    return fig