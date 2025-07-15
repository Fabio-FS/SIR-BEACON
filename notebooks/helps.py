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

def create_subplot_grid(
    results_list,
    plot_type="homophily",  # or "polarization"
    values_list=None,
    model_list=None,
    param_list=None,
    nrows=1,
    ncols=1,
    figsize=(10, 8),
    save_path=None
):
    """
    Create a grid of subplots using either plot_fixed_homophily or plot_fixed_polarization
    
    Args:
        results_list: List of results dictionaries
        plot_type: Type of plot for each subplot, either "homophily" or "polarization"
        values_list: List of values to use for each subplot (e.g., [0,3,6] for homophily)
        model_list: List of model modules to use for each subplot
        param_list: List of parameter dictionaries to use for each subplot
        nrows, ncols: Grid dimensions
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Figure with grid of plots
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec
    import os
    
    # Validate plot_type
    if plot_type not in ["homophily", "polarization"]:
        raise ValueError("plot_type must be either 'homophily' or 'polarization'")
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows, ncols, figure=fig)
    
    # Create temporary directory for individual plots if it doesn't exist
    temp_dir = "temp_plots"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # Default values if not provided
    if values_list is None:
        if plot_type == "homophily":
            values_list = [[0, 3, 6]] * (nrows * ncols)
        else:  # polarization
            values_list = [[0, 0.5, 1]] * (nrows * ncols)
    
    # If only one set of values provided, replicate for all subplots
    if len(values_list) == 1 and nrows * ncols > 1:
        values_list = values_list * (nrows * ncols)
    
    # Ensure the lists have the right length
    if len(results_list) < nrows * ncols:
        print(f"Warning: Not enough results provided ({len(results_list)}) for grid size ({nrows}x{ncols})")
    
    if model_list is None or len(model_list) < nrows * ncols:
        if model_list is None:
            model_list = []
        # Use the first model for all if not enough provided
        model = model_list[0] if model_list else None
        model_list = [model] * (nrows * ncols)
    
    if param_list is None or len(param_list) < nrows * ncols:
        if param_list is None:
            param_list = []
        # Use the first params for all if not enough provided
        params = param_list[0] if param_list else {}
        param_list = [params] * (nrows * ncols)
    
    # Create each subplot
    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            
            if idx < len(results_list):
                ax = fig.add_subplot(gs[i, j])
                
                # Create temporary filename for this subplot
                temp_file = f"{temp_dir}/subplot_{i}_{j}.png"
                
                # Create the plot
                if plot_type == "homophily":
                    plot_fixed_homophily(
                        results=results_list[idx],
                        H=values_list[idx],
                        model=model_list[idx],
                        PARAM=param_list[idx],
                        name=temp_file
                    )
                else:  # polarization
                    plot_fixed_polarization(
                        results=results_list[idx],
                        P=values_list[idx],
                        model=model_list[idx],
                        PARAM=param_list[idx],
                        name=temp_file
                    )
                
                # Read the created image and display in the subplot
                img = plt.imread(temp_file)
                ax.imshow(img)
                ax.axis('off')  # Hide axis
            else:
                # Hide unused subplots
                ax = fig.add_subplot(gs[i, j])
                ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    
    # Clean up temporary files
    for i in range(nrows):
        for j in range(ncols):
            temp_file = f"{temp_dir}/subplot_{i}_{j}.png"
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
        os.rmdir(temp_dir)
    
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