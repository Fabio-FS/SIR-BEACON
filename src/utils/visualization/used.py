import matplotlib.pyplot as plt
import json
import ast

from src.models import SIRM, SIRT, SIRV, SIRM_isolated, SIRV_isolated, SIRT_isolated
from src.utils.batch_sweep import sweep_one_parameter, sweep_two_parameters
from src.utils.Contact_Matrix import create_contact_matrix
from ..distributions import pol_mean_to_ab
#from src.utils.visualization.heatmaps import plot_compartment_heatmap2
#from src.utils.visualization.core import create_standalone_colorbar, Lx, Ly, discretize_cmaps

# Import the matrix creation function

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from scipy.stats import beta
import numpy as np
import jax.numpy as jnp

from typing import List, Optional, Tuple, Dict, Any
from tqdm import tqdm  # Optional: for progress tracking

import matplotlib.patches as patches

import pandas as pd

Lx = 2.29
Ly = 2.16











def generate_colorbar_from_color_list(colors, pathname):
    # Create a custom colormap using your colors
    cmap_name = 'custom_diverging'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 1))

    # Create data for the colorbar
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    # Display the colorbar
    ax.imshow(gradient, aspect='auto', cmap=custom_cmap)

    # Remove ticks from the plot
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 256, 3))
    ax.set_xticklabels([])


    # Optional: Add a border to the colorbar
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.5)
    
    fig.savefig(pathname, dpi=300)
    return fig, custom_cmap

def plot_beta_with_gradient(alpha, beta_param, custom_cmap , Nbins=100):
    fig, ax = plt.subplots(figsize=(Lx/2*5, Ly/3*5))
    
    x = np.linspace(1/Nbins/2, 1-1/Nbins/2, Nbins)
    y = beta.pdf(x, alpha, beta_param)
    ax.plot(x, y, '--', color='black', linewidth=5)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    lc = LineCollection(segments, cmap=custom_cmap, norm=plt.Normalize(0, 1))
    lc.set_array(x[:-1])
    lc.set_linewidth(20)
    
    line = ax.add_collection(lc)
       
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.05, 5.05)
    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    fig.patch.set_visible(False)
    plt.show()

    return fig, ax


def plot_contact_matrix(h_value: float, 
                        n_groups: int = 5,
                        figsize: Tuple[int, int] = (6, 5),
                        cmap: str = "Blues",
                        save_path: Optional[str] = None):
    """
    Plot a single contact matrix for a specific homophilic tendency (h) value.
    
    Args:
        h_value: Homophilic tendency value to visualize
        n_groups: Number of population groups (matrix size will be n_groups x n_groups)
        figsize: Figure size (width, height)
        cmap: Colormap to use
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure object
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create an equal population distribution (a = b = 1)
    pop = jnp.ones(n_groups) * 0.2
    
    # Create contact matrix
    C = create_contact_matrix(n_groups, h_value, pop)
    C = np.flipud(C)
    
    # Plot as heatmap
    ax.imshow(C, cmap=cmap, vmin=0, vmax=3)
    
    # Add text annotations to each cell
    for row in range(n_groups):
        for col in range(n_groups):
            value = np.round(C[row, col], 1)
            text_color = "black" if value < 1.5 else "white"
            ax.text(col, row, f"{value:.1f}", 
                    ha="center", va="center", 
                    color=text_color, fontsize=9)
    
    # Clean up the display
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    
    # Optional title
    # ax.set_title(f"Contact Matrix (h = {h_value})")
    
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    return fig


def create_standalone_colorbar(cmap_name, vmin, vmax, height, width, orientation='vertical', 
                              title=None, ticks=None, ticklabels=None, save_path=None):
    """
    Create a standalone colorbar with specific dimensions.
    
    Args:
        cmap_name: Name of the colormap (e.g., 'Blues_r', 'hot_r', 'viridis_r')
        vmin, vmax: Min and max values for the colorbar
        height: Height of the colorbar in inches
        width: Width of the colorbar in inches
        orientation: 'vertical' or 'horizontal'
        title: Optional title for the colorbar
        ticks: Optional list of tick positions
        ticklabels: Optional list of tick labels
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure object
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    # Create a figure with the specified dimensions
    fig = plt.figure(figsize=(width, height))
    
    # Create a dummy ScalarMappable to use with the colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_name),
                              norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    
    # Create axes for the colorbar
    # For vertical orientation, use a narrow width; for horizontal, use a narrow height
    if orientation == 'vertical':
        cax = fig.add_axes([0.3, 0.1, 0.2, 0.8])  # [left, bottom, width, height]
    else:
        cax = fig.add_axes([0.1, 0.4, 0.8, 0.2])  # [left, bottom, width, height]
    
    # Create the colorbar
    cbar = plt.colorbar(sm, cax=cax, orientation=orientation)
    
    # Add title if provided
    if title:
        cbar.set_label(title)
    
    # Set custom ticks if provided
    if ticks is not None:
        cbar.set_ticks(ticks)
    
    # Set custom tick labels if provided
    if ticklabels is not None:
        if orientation == 'vertical':
            cbar.ax.set_yticklabels(ticklabels)
        else:
            cbar.ax.set_xticklabels(ticklabels)
    
    # Remove unnecessary white space
    fig.patch.set_visible(False)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, transparent=True)
    
    return fig

######################################################
######################################################
######################################################
######################################################


def discretize_cmaps(name: str, N: int) -> ListedColormap:
    """
    Create a discretized version of a colormap
    
    Args:
        name: Name of the matplotlib colormap
        N: Number of discrete colors
        
    Returns:
        Discretized colormap
    """
    c_map = plt.colormaps[name]
    colors = c_map(np.linspace(0, 1, N))
    res = ListedColormap(colors)
    return res


my_hot_r = discretize_cmaps('hot_r', 12)
my_hot_r.set_bad('gray')

# my_vir_r is used for R0 heatmaps
my_vir_r = discretize_cmaps('viridis_r', 12)
my_vir_r.set_bad('gray')

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







def prepare_matrix_data(results: Dict[str, Any], metric: str) -> np.ndarray:
    """
    Extract a simple matrix from results for plotting.
    
    Args:
        results: Dictionary returned by sweep_two_parameters
        metric: Metric to extract ("infections", "r0", or compartment names)
    
    Returns:
        numpy array containing the data matrix
    """
    if metric == "infections":
        # Total infections = 1 - final susceptible - final vaccinated
        data = np.ones_like(results['r0'])
        
        # Subtract final susceptible population
        if "S" in results['final_state']:
            data -= np.array(jnp.sum(results['final_state']["S"], axis=2))
            
        # Subtract final vaccinated population if it exists
        if "V" in results['final_state']:
            data -= np.array(jnp.sum(results['final_state']["V"], axis=2))
            
    elif metric == "r0":
        data = np.array(results['r0'])
        
    elif metric in results['final_state']:
        # Extract a specific compartment
        data = np.array(jnp.sum(results['final_state'][metric], axis=2))
        
    else:
        raise ValueError(f"Unknown metric: {metric}")
        
    return data


def extract_plot_parameters(results: Dict[str, Any], metric: str) -> Dict[str, Any]:
    """
    Extract parameters needed for plotting from results.
    
    Args:
        results: Dictionary returned by sweep_two_parameters
        metric: Metric to extract ("infections", "r0", or compartment names)
    
    Returns:
        Dictionary with plot parameters:
        - x_values: Values for x-axis
        - y_values: Values for y-axis  
        - x_label: Label for x-axis
        - y_label: Label for y-axis
        - title: Plot title
        - cbar_label: Label for colorbar
        - vmin, vmax: Suggested data range
    """
    # Extract parameter values and names
    param1_vals = results['parameter_grid']['param1_vals'][0]  # First row
    param2_vals = results['parameter_grid']['param2_vals'][:, 0]  # First column
    param1_name = results['parameter_names']['param1']
    param2_name = results['parameter_names']['param2']
    
    # Create plot parameters dictionary
    plot_params = {
        'x_values': param1_vals,
        'y_values': param2_vals
    }
    
    # Set metric-specific parameters
    if metric == "infections":
        plot_params.update({
            'vmin': 0,
            'vmax': 1
        })
    elif metric == "r0":
        plot_params.update({
            
            'vmin': 1,
            'vmax': 10
        })
    
    
    return plot_params


def plot_matrix(
    data: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 8),
    contour_values: Optional[List[float]] = None,
    contour_colors: Optional[List[str]] = None,
    xticks: Optional[List[float]] = None,
    yticks: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    rect_coords: Optional[List[float]] = None
) -> plt.Figure:
    """
    Plot a matrix using imshow.
    
    Args:
        data: Matrix to plot
        x_values: Values for the x-axis
        y_values: Values for the y-axis
        cmap: Colormap to use
        vmin: Minimum value for colorscale
        vmax: Maximum value for colorscale
        figsize: Figure size (width, height)
        contour_values: Optional list of values for contour lines
        contour_colors: Optional list of colors for contour lines
        xticks: Optional list of tick positions for x-axis
        yticks: Optional list of tick positions for y-axis
        save_path: Path to save the figure (if None, figure is not saved)
        rect_coords: Optional rectangle coordinates [x, y, width, height]
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate extents for proper alignment
    x_min, x_max = x_values.min(), x_values.max()
    y_min, y_max = y_values.min(), y_values.max()
    
    # Calculate step sizes
    x_step = (x_max - x_min) / (len(x_values) - 1)
    y_step = (y_max - y_min) / (len(y_values) - 1)
    
    # Set extent to align with parameter values
    extent = [
        x_min - x_step/2,  # left
        x_max + x_step/2,  # right
        y_min - y_step/2,  # bottom
        y_max + y_step/2   # top
    ]
    
    # Create heatmap
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, 
              extent=extent, origin='lower', aspect='auto')
    
    # Add contours if specified
    if contour_values is not None:
        # Create coordinate grids for contour
        X, Y = np.meshgrid(
            np.linspace(x_min, x_max, data.shape[1]), 
            np.linspace(y_min, y_max, data.shape[0])
        )
        if contour_colors is None:
            contour_colors = ["black"] * len(contour_values)
            
        contour = ax.contour(X, Y, data, levels=contour_values, 
                            colors=contour_colors, linewidths=1.5)
        ax.clabel(contour, inline=True, fontsize=8)
    
    # Set ticks if provided
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels([])
    
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels([])
    
    # Add rectangle if coordinates provided
    if rect_coords is not None:
        # Import matplotlib patches if not already imported
        import matplotlib.patches as patches
        
        # rect_coords should be [x, y, width, height]
        rect = patches.Rectangle(
            (rect_coords[0], rect_coords[1]),  # (x, y)
            rect_coords[2],                   # width
            rect_coords[3],                   # height
            linewidth=2,
            edgecolor='black',
            facecolor='none'
        )
        ax.add_patch(rect)
    
    fig.patch.set_visible(False)  # Make figure background transparent
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    return fig


def analyze_matrix(data: np.ndarray, metric: str) -> np.ndarray:
    matrix_data = prepare_matrix_data(data, metric)
    plot_params = extract_plot_parameters(data, metric)
    return matrix_data, plot_params


def from_mm_to_in(value_in_mm):
    """Convert millimeters to inches.
    
    Args:
        value_in_mm: Value in millimeters
        
    Returns:
        float: Value converted to inches
    """
    return value_in_mm / 25.4




def plot_slices(
    model_module: Optional[Any],
    matrix_data: np.ndarray,
    plot_params: Dict[str, Any],
    model_params: Dict[str, Any],
    slice_values: List[float],
    slice_dimension: str = "y",
    baseline_compartment: str = 'R',
    colors: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (2.29, 2.16),
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    output_path: Optional[str] = None,
    population_size: int = 5
) -> plt.Figure:
    """
    Plot slices from a 2D parameter sweep at specified values.
    
    Args:
        matrix_data: 2D numpy array with simulation results
        plot_params: Dictionary with x_values and y_values arrays
        slice_values: Values at which to take slices
        slice_dimension: 'x' for fixed x-values (columns), 'y' for fixed y-values (rows)
        baseline_compartment: Compartment to measure for baseline ('R', 'S', etc.)
        colors: List of colors for each slice
        figsize: Figure size as (width, height)
        xlim: x-axis limits as (min, max)
        ylim: y-axis limits as (min, max)
        output_path: Path to save the figure
        population_size: Population size for baseline calculation
    
    Returns:
        Matplotlib figure object
    """
    # Get axis values
    x_values = plot_params['x_values']
    y_values = plot_params['y_values']
    
    # Calculate baseline if requested
    baseline_value = None
        
    # Set fixed mean if not already present
    params_copy = model_params.copy()
    if "fixed_mean" not in params_copy:
        params_copy["fixed_mean"] = 0.5
    
    # Run a simple simulation with near-consensus
    baseline_result = sweep_two_parameters(
        model_module=model_module,
        param1_name='beta_params',
        param1_range=[0.000001],  # Near-zero polarization for consensus
        param2_name='homophilic_tendency',
        param2_range=[0],
        custom_base_params=params_copy,
        population_size=population_size,
    )
    
    # Calculate the baseline value
    baseline_value = np.sum(baseline_result['final_state'][baseline_compartment][0, 0])

    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    

    
    # Plot slices
    for i, value in enumerate(slice_values):
        color = colors[i]
            
        if slice_dimension == 'y':
            # Fixed y-value (row) - find the closest index
            idx = np.argmin(np.abs(y_values - value))
            ax.plot(x_values, matrix_data[idx, :], color=color, linewidth=2)
        else:
            # Fixed x-value (column) - find the closest index
            idx = np.argmin(np.abs(x_values - value))
            ax.plot(y_values, matrix_data[:, idx], color=color, linewidth=2)

    if slice_dimension == 'y':
        ax.plot([x_values[0], x_values[-1]], [baseline_value, baseline_value],                                               
                color='black', linestyle='--')
    else:
        ax.plot([y_values[0], y_values[-1]], [baseline_value, baseline_value],                                               
                color='black', linestyle='--')
    
    # Set axis limits
    if xlim is None:
        if slice_dimension == 'y':
            xlim = (x_values[0], x_values[-1])
        else:
            xlim = (y_values[0], y_values[-1])
    
    if ylim is None:
        ylim = (0, 1)  # Default for proportion data
        
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Set minimal ticks (start, middle, end)
    if slice_dimension == 'y':
        ax.set_xticks([x_values[0], x_values[len(x_values)//2], x_values[-1]])
    else:
        ax.set_xticks([y_values[0], y_values[len(y_values)//2], y_values[-1]])
    
    ax.set_yticks([0, 0.5, 1])
    
    # Remove tick labels for minimal design
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Clean up the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Make figure background transparent
    fig.patch.set_visible(False)
    
    # Save if path provided
    if output_path:
        plt.savefig(output_path)
    
    return fig

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


def plot_double_comparison(days, mins, maxs, bases, OG, pathname, Lx, Ly, x_max = 600, y_max = 0.6):
    fig, ax = plt.subplots(1, 1, figsize=(Lx, Ly))

    Im, Rm = mins[0], mins[1]
    IM, RM = maxs[0], maxs[1]

    Ib, Rb = bases[0], bases[1]     # no info about distribution of behavior only average, beta and gamma and range
    IG, RG = OG[0], OG[1]           # no info about average behavior, only about beta and gamma and range
    
    ax.fill_between(days, Rm+Im, RM+IM, color='#b3de69', alpha=1)
    ax.plot(days, Rm+Im, color ="black", linewidth=0.5)
    ax.plot(days, RM+IM, color ="black", linewidth=0.5)
    ax.plot(days, Rb+Ib, '--',color ="black")

    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.01, y_max)

    ax.set_xticks([0,  x_max/2, x_max])
    ax.set_yticks([0,  y_max/2, y_max])

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # make background transparent
    fig.patch.set_visible(False)  # Make figure background transparent

    # save figure
    plt.savefig(pathname)
    return fig

def run_single_simulation(polarization, homophilic_tendency, fixed_mean, PARAMS, model, simulated_days=1000,
                          initial_infected_prop = 1e-4, population_size = 5):

    # Set up parameters for this simulation
    params = PARAMS.copy()
    params.update({
        'homophilic_tendency': homophilic_tendency,
        'fixed_mean': fixed_mean
    })
    
    # Calculate beta parameters from polarization and mean
    beta_params = pol_mean_to_ab(polarization, params['fixed_mean'])
    
    # Run the simulation
    states_trajectory, r0, homophily = model.run_simulation(
        beta_params=beta_params,
        params=params,
        simulated_days=simulated_days,
        initial_infected_prop=initial_infected_prop,
        population_size=population_size,
        use_contact_matrix=True,
        return_trajectory=True
    )
    
    # Unpack and sum the state trajectories across population compartments
    S, I, R, *_ = states_trajectory
    S_total = jnp.sum(S, axis=1)
    I_total = jnp.sum(I, axis=1)
    R_total = jnp.sum(R, axis=1)
    
    return S_total, I_total, R_total

def get_compartment_temporal_data(
    polarization: float,
    homophilic_tendency: float,
    fixed_mean: float,
    PARAMS: Dict[str, Any],
    model: Any,
    simulated_days: int = 1000,
    initial_infected_prop: float = 1e-4,
    population_size: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate temporal data for I+R compartments for each population group.
    
    Args:
        polarization: Polarization parameter
        homophilic_tendency: Homophilic tendency parameter
        fixed_mean: Fixed mean parameter
        PARAMS: Dictionary of model parameters
        model: Model class to use
        simulated_days: Number of days to simulate
        initial_infected_prop: Initial infected proportion
        population_size: Size of the population
        
    Returns:
        Tuple containing:
        - IR: Array of shape (simulated_days, population_size) with I+R fractions
        - time_points: Array of time points
    """
    # Run simulation to get the data
    states_trajectory, _, _ = model.run_simulation(
        beta_params=pol_mean_to_ab(polarization, fixed_mean),
        params={**PARAMS, 'homophilic_tendency': homophilic_tendency, 'fixed_mean': fixed_mean},
        simulated_days=simulated_days,
        initial_infected_prop=initial_infected_prop,
        population_size=population_size,
        use_contact_matrix=True,
        return_trajectory=True
    )
    
    time_points = np.arange(simulated_days)
    # Extract I and R compartments
    if model == SIRV:
        S, I, R, V = states_trajectory
        return S, I, R, V, time_points
    else:
        S, I, R = states_trajectory
        return S, I, R, time_points
    
    # Create time points array
    
    IR = (I + R) / (S[0] + I[0] + R[0] + V[0])
    IR = (I + R) / (S[0] + I[0] + R[0])

def plot_compartment_heatmap_temporal(
    IR: np.ndarray,
    time_points: np.ndarray,
    population_size: int = 5,
    figsize: Tuple[float, float] = (Lx, Ly),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a heatmap showing the temporal evolution of I+R compartments for each population group.
    
    Args:
        IR: Array of shape (simulated_days, population_size) with I+R fractions
        time_points: Array of time points
        population_size: Size of the population
        figsize: Figure size
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(IR.T,  # Transpose to have time on x-axis and groups on y-axis
                  cmap=my_hot_r,
                  aspect='auto',
                  origin='lower',
                  extent=[time_points[0], time_points[-1], 0, population_size],
                  interpolation='none')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('I+R Fraction')
    
    # Set labels
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Population Group')
    
    # Set ticks
    ax.set_yticks(np.arange(population_size) + 0.5)
    ax.set_yticklabels([f'Group {i+1}' for i in range(population_size)])
    
    # Make figure background transparent
    fig.patch.set_visible(False)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    return fig




# ----------------- For Figure 3 (experimental) -----------------

def extract_behavior_distribution(df, behavior_type):
    df = df.dropna()
    
    # Extract self-reported behavior
    self_behavior = df[f'{behavior_type}_self'].astype(int)
    
    # Count occurrences of each behavior level (1-5)
    n_bins = 5
    behavior_vector = np.zeros(n_bins)
    for i in range(n_bins):
        behavior_vector[i] = np.sum(self_behavior == i+1)
    
    # Normalize to get distribution
    behavior_vector = behavior_vector / np.sum(behavior_vector)
    
    return behavior_vector
def generate_raw_matrix(df, behavior_type = "masks"):
    n_bins = 5
    df_clean = df.dropna()
    contact_matrix = np.zeros((n_bins, n_bins))
    
    # Define columns based on behavior type
    if behavior_type == "vacc":
        cols = [f'{behavior_type}_others0{i+1}' for i in range(n_bins)]
    else:
        cols = [
            f'{behavior_type}_others_never', 
            f'{behavior_type}_others_sometimes',
            f'{behavior_type}_others_half',
            f'{behavior_type}_others_often',
            f'{behavior_type}_others_always'
        ]
    for i, row in df.iterrows():
        self_idx = int(row[f'{behavior_type}_self']) - 1
        for j, col in enumerate(cols):
            if pd.notna(row[col]):
                contact_matrix[self_idx, j] += row[col]
    
    return contact_matrix

def generate_contact_matrix(df, behavior_type = "masks"):
    behavior_distribution  = extract_behavior_distribution(df, behavior_type)
    raw_matrix = generate_raw_matrix(df, behavior_type)
    contact_matrix = create_contact_matrix(5, None, behavior_distribution, Cm_external=raw_matrix)
    return contact_matrix



def plot_histogram_distribution(distribution, figsize=(Lx, Ly), save_path=None):
    colors = ['#276419', '#7FBC41', '#F7F7F7', '#DE77AE', '#8E0152']
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(1, 6)  # Categories 1-5
    distribution = np.flipud(distribution)
    for i in range(5):
        ax.bar(x[i], distribution[i], width=1.0, color=colors[i], edgecolor='black', linewidth=1)    
    ax.set_ylim(0, 0.5)

    ax.set_xticks([1,2,3,4,5])
    ax.set_xticklabels([])

    ax.set_yticks([0,0.25,0.5])
    ax.set_yticklabels([])

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    
    # remove top and left spines
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)    
    # Set background to transparent
    fig.patch.set_alpha(0.0)
    fig.patch.set_visible(False)
    
    # Save figure if path is provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_contact_matrix(matrix,  Lx, Ly, path = None):
    fig, ax = plt.subplots(figsize=(Lx, Ly))
    ax.imshow(np.flipud(matrix), cmap="Blues", interpolation='nearest', vmin=0, vmax=2.6)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.patch.set_alpha(0.0)
    fig.patch.set_visible(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # Get value from the original matrix (not the flipped one)
            value = np.round(matrix[matrix.shape[0]-1-i, j],1)
            # Add text annotation
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", 
                   color="black" if value < 0.5*np.max(matrix) else "white")
    if path:
        fig.savefig(path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def bootstrap_pol_mean(df, behavior_type, n_bootstrap=1000, seed=42):

    np.random.seed(seed)
    
    # Get clean dataset with no NAs in the target behavior
    df_clean = df.dropna(subset=[f'{behavior_type}_self'])
    n_samples = len(df_clean)
    
    # Arrays to store bootstrap results
    bootstrap_means = np.zeros(n_bootstrap)
    bootstrap_polarizations = np.zeros(n_bootstrap)
    
    # Perform bootstrap
    for i in tqdm(range(n_bootstrap), desc=f"Bootstrapping {behavior_type}"):
        # Sample with replacement
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_sample = df_clean.iloc[bootstrap_indices]
        
        # Get self-reported behavior and rescale from 1-5 to 0-1
        self_behavior = bootstrap_sample[f'{behavior_type}_self'].astype(int)
        rescaled_behavior = (self_behavior - 1) / 4  # Rescale from 1-5 to 0-1
        
        # Store results
        bootstrap_means[i] = np.mean(rescaled_behavior)
        bootstrap_polarizations[i] = 4 *np.var(rescaled_behavior)
    
    # Calculate summary statistics
    mean_estimate = np.mean(bootstrap_means)
    polarization_estimate = np.mean(bootstrap_polarizations)
    
    # Calculate 95% confidence intervals
    mean_ci = np.percentile(bootstrap_means, [2.5, 97.5])
    polarization_ci = np.percentile(bootstrap_polarizations, [2.5, 97.5])
    
    return {
        'bootstrap_means': bootstrap_means,
        'bootstrap_polarizations': bootstrap_polarizations,
        'mean_estimate': mean_estimate,
        'mean_ci': mean_ci,
        'polarization_estimate': polarization_estimate,
        'polarization_ci': polarization_ci
    }



def create_combined_behaviors_plot(results_dict, figsize=(16, 6), cmap='viridis', 
                                  bins=30, highlight_color='red'):
    """
    Create a combined figure showing joint plots for multiple behaviors.
    
    Args:
        results_dict: Dictionary where keys are behavior types and values are dictionaries with:
                     - 'bootstrap_means': Array of bootstrap means
                     - 'bootstrap_polarizations': Array of bootstrap polarizations
                     - 'mean_estimate': Point estimate for mean
                     - 'mean_ci': Confidence interval for mean [lower, upper]
                     - 'polarization_estimate': Point estimate for polarization
                     - 'polarization_ci': Confidence interval for polarization [lower, upper]
        figsize: Size of the figure (width, height) in inches
        cmap: Colormap for the 2D histograms
        bins: Number of bins for histograms
        highlight_color: Color for the central points and reference lines
        
    Returns:
        fig: The matplotlib figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.colors import LogNorm
    
    # Get behavior types
    behavior_types = list(results_dict.keys())
    n_behaviors = len(behavior_types)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Set up main grid: 3 columns (one for each behavior)
    outer_grid = GridSpec(1, n_behaviors, figure=fig, wspace=0.3)
    
    for i, behavior in enumerate(behavior_types):
        # Get data for this behavior
        result = results_dict[behavior]
        
        # Prepare data
        means = result['bootstrap_means']
        polarizations = result['bootstrap_polarizations']
        mean_estimate = result['mean_estimate']
        pol_estimate = result['polarization_estimate']
        mean_ci = result['mean_ci']
        pol_ci = result['polarization_ci']
        
        # Create nested grid for this behavior plot (4x4 for joint plot + marginals)
        inner_grid = GridSpecFromSubplotSpec(4, 4, subplot_spec=outer_grid[i], 
                                           wspace=0.05, hspace=0.05)
        
        # Create axes
        ax_joint = fig.add_subplot(inner_grid[1:, 0:3])
        ax_marg_x = fig.add_subplot(inner_grid[0, 0:3], sharex=ax_joint)
        ax_marg_y = fig.add_subplot(inner_grid[1:, 3], sharey=ax_joint)
        
        # Plot joint distribution with normalized histogram
        H, xedges, yedges = np.histogram2d(means, polarizations, bins=bins)
        H = H / len(means)  # Normalize without factor of 3
        
        X, Y = np.meshgrid((xedges[:-1] + xedges[1:])/2, (yedges[:-1] + yedges[1:])/2)
        
        # Plot the 2D histogram using pcolormesh for better control
        im = ax_joint.pcolormesh(xedges, yedges, H.T, cmap=cmap, norm=LogNorm())
        
        # Add contour lines
        #ax_joint.contour(X, Y, H.T, levels=5, colors='white', alpha=0.5)
        
        # Plot marginal distributions with normalized histograms
        weights = np.ones_like(means) / len(means)
        ax_marg_x.hist(means, bins=bins, color='blue', alpha=0.7, weights=weights)
        ax_marg_y.hist(polarizations, bins=bins, orientation='horizontal', 
                       color='green', alpha=0.7, weights=weights)
        
        # Add highlight for central estimate
        ax_joint.scatter([mean_estimate], [pol_estimate], color=highlight_color, 
                         marker='x', s=100, zorder=5)
        
        # Add reference lines
        ax_joint.axvline(mean_estimate, color=highlight_color, linestyle='--', alpha=0.5)
        ax_joint.axhline(pol_estimate, color=highlight_color, linestyle='--', alpha=0.5)
        
        # Add confidence interval lines
        ax_marg_x.axvline(mean_estimate, color=highlight_color, linestyle='--')
        ax_marg_x.axvline(mean_ci[0], color=highlight_color, linestyle=':', alpha=0.7)
        ax_marg_x.axvline(mean_ci[1], color=highlight_color, linestyle=':', alpha=0.7)
        
        ax_marg_y.axhline(pol_estimate, color=highlight_color, linestyle='--')
        ax_marg_y.axhline(pol_ci[0], color=highlight_color, linestyle=':', alpha=0.7)
        ax_marg_y.axhline(pol_ci[1], color=highlight_color, linestyle=':', alpha=0.7)
        
        # Remove tick labels from marginal plots
        ax_marg_x.tick_params(axis='x', labelbottom=False)
        ax_marg_y.tick_params(axis='y', labelleft=False)
        
        # Set titles and labels
        behavior_label = behavior.capitalize()
        if behavior == "vacc":
            behavior_label = "Vaccination"
        elif behavior == "testing":
            behavior_label = "Testing"
        ax_marg_x.set_title(f"{behavior_label}")
        
        # Only add y-label to the first plot
        if i == 0:
            ax_joint.set_ylabel('Polarization (4 × variance)')
        
        # Add x-label to all plots
        ax_joint.set_xlabel('Mean (0-1 scale)')
        
        # Add colorbar only to the last plot
        if i == n_behaviors - 1:
            # Create custom axes for the colorbar that matches the exact height of the right histogram
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax_marg_y)
            cax = divider.append_axes("right", size="20%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('Fraction of occurence')
    
    plt.tight_layout()
    return fig



def merge_behavior_datasets(results_dict):
    """
    Merge multiple behavior datasets into a single dataset with source labels.
    
    Args:
        results_dict: Dictionary where keys are behavior types and values are dictionaries with:
                     - 'bootstrap_means': Array of bootstrap means
                     - 'bootstrap_polarizations': Array of bootstrap polarizations
                     - 'mean_estimate': Point estimate for mean
                     - 'mean_ci': Confidence interval for mean [lower, upper]
                     - 'polarization_estimate': Point estimate for polarization
                     - 'polarization_ci': Confidence interval for polarization [lower, upper]
                     
    Returns:
        Dictionary with:
        - 'means': Combined array of all bootstrap means
        - 'polarizations': Combined array of all bootstrap polarizations
        - 'sources': Array of source labels (index of the behavior type)
        - 'behavior_types': List of behavior type names
        - 'point_estimates': Dictionary with mean and polarization estimates for each behavior
    """
    import numpy as np
    
    # Get behavior types
    behavior_types = list(results_dict.keys())
    n_behaviors = len(behavior_types)
    
    # Initialize empty lists for combined data
    all_means = []
    all_polarizations = []
    all_sources = []
    
    # Store point estimates
    point_estimates = {
        'means': [],
        'mean_cis': [],
        'polarizations': [],
        'polarization_cis': []
    }
    
    # Merge datasets
    for i, behavior in enumerate(behavior_types):
        # Get data
        result = results_dict[behavior]
        means = result['bootstrap_means']
        polarizations = result['bootstrap_polarizations']
        
        # Append data
        all_means.extend(means)
        all_polarizations.extend(polarizations)
        
        # Create and append source labels
        sources = np.full(len(means), i)
        all_sources.extend(sources)
        
        # Store point estimates
        point_estimates['means'].append(result['mean_estimate'])
        point_estimates['mean_cis'].append(result['mean_ci'])
        point_estimates['polarizations'].append(result['polarization_estimate'])
        point_estimates['polarization_cis'].append(result['polarization_ci'])
    
    # Convert lists to numpy arrays
    merged_data = {
        'means': np.array(all_means),
        'polarizations': np.array(all_polarizations),
        'sources': np.array(all_sources),
        'behavior_types': behavior_types,
        'point_estimates': point_estimates
    }
    
    return merged_data


def plot_merged_behaviors(merged_data, figsize=(10, 8), bins=150, 
                         dataset_colors=['blue', 'green', 'red'], 
                         highlight_colors=['darkblue', 'darkgreen', 'darkred'],
                         alpha=0.7, show_combined_heatmap=True):
    """
    Create a plot showing the merged behaviors dataset with each source colored differently.
    
    Args:
        merged_data: Output from merge_behavior_datasets function
        figsize: Size of the figure (width, height) in inches
        bins: Number of bins for histograms
        dataset_colors: List of colors for each dataset
        highlight_colors: List of colors for the central points and reference lines
        alpha: Transparency for overlapping elements
        show_combined_heatmap: Whether to show a combined heatmap or separate by source
        
    Returns:
        fig: The matplotlib figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # Extract data
    means = merged_data['means']
    polarizations = merged_data['polarizations']
    sources = merged_data['sources']
    behavior_types = merged_data['behavior_types']
    point_estimates = merged_data['point_estimates']
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    grid = GridSpec(4, 4, figure=fig, wspace=0.05, hspace=0.05)
    
    # Create axes
    ax_joint = fig.add_subplot(grid[1:, 0:3])
    ax_marg_x = fig.add_subplot(grid[0, 0:3], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(grid[1:, 3], sharey=ax_joint)
    
    # Determine data range with padding
    min_mean, max_mean = np.min(means), np.max(means)
    min_pol, max_pol = np.min(polarizations), np.max(polarizations)
    mean_padding = (max_mean - min_mean) * 0.05
    pol_padding = (max_pol - min_pol) * 0.05
    
    # Create bins
    mean_bins = np.linspace(min_mean - mean_padding, max_mean + mean_padding, bins+1)
    pol_bins = np.linspace(min_pol - pol_padding, max_pol + pol_padding, bins+1)
    
    # Main plot: combined heatmap
    H, xedges, yedges = np.histogram2d(means, polarizations, bins=[mean_bins, pol_bins])
    H = H * 3 / len(means)  # Scale the histogram
    
    # Plot heatmap and contours
    im = ax_joint.pcolormesh(xedges, yedges, H.T, cmap='viridis', norm=LogNorm())
    X, Y = np.meshgrid((xedges[:-1] + xedges[1:])/2, (yedges[:-1] + yedges[1:])/2)
    #ax_joint.contour(X, Y, H.T, levels=5, colors='white', alpha=0.7)
    
    # Add colorbar
    divider = make_axes_locatable(ax_marg_y)
    cax = divider.append_axes("right", size="20%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Fraction of occurrences')
    
    # Plot marginals and references for each behavior type
    for i, behavior in enumerate(behavior_types):
        mask = (sources == i)
        source_means = means[mask]
        source_pols = polarizations[mask]
        color = dataset_colors[i % len(dataset_colors)]
        highlight = highlight_colors[i % len(highlight_colors)]
        
        # Plot histograms
        weights = np.ones_like(source_means) / len(source_means) * 3.0
        ax_marg_x.hist(source_means, bins=mean_bins, color=color, alpha=alpha, 
                       label=f"{behavior.capitalize()} (n={len(source_means)})", weights=weights)
        ax_marg_y.hist(source_pols, bins=pol_bins, orientation='horizontal', 
                       color=color, alpha=alpha, weights=weights)
        
        # Add point estimates and reference lines
        mean_val = point_estimates['means'][i]
        pol_val = point_estimates['polarizations'][i]
        mean_ci = point_estimates['mean_cis'][i]
        pol_ci = point_estimates['polarization_cis'][i]
        
        # Central point
        ax_joint.scatter([mean_val], [pol_val], color=highlight, marker='x', s=100, zorder=10)
        
        # Reference lines
        ax_joint.axvline(mean_val, color=highlight, linestyle='--', alpha=0.5)
        ax_joint.axhline(pol_val, color=highlight, linestyle='--', alpha=0.5)
        
        # Confidence intervals in marginals
        ax_marg_x.axvline(mean_val, color=highlight, linestyle='--')
        ax_marg_x.axvline(mean_ci[0], color=highlight, linestyle=':', alpha=0.7)
        ax_marg_x.axvline(mean_ci[1], color=highlight, linestyle=':', alpha=0.7)
        
        ax_marg_y.axhline(pol_val, color=highlight, linestyle='--')
        ax_marg_y.axhline(pol_ci[0], color=highlight, linestyle=':', alpha=0.7)
        ax_marg_y.axhline(pol_ci[1], color=highlight, linestyle=':', alpha=0.7)
    
    # Finalize plots
    ax_marg_x.tick_params(axis='x', labelbottom=False)
    ax_marg_y.tick_params(axis='y', labelleft=False)
    
    ax_marg_x.set_title("Combined Behaviors")
    ax_joint.set_ylabel('Polarization (4 × variance)')
    ax_joint.set_xlabel('Mean (0-1 scale)')
    
    ax_joint.legend(fontsize='small', loc='upper right')
    
    plt.tight_layout()
    return fig



def generate_raw_matrix_single_agent(df, k, behavior_type="masks"):
    n_bins = 5
    contact_matrix = np.zeros((n_bins, n_bins))
    
    # Get data for agent k only
    row = df.iloc[k]
    
    # Define columns based on behavior type
    if behavior_type == "vacc":
        cols = [f'{behavior_type}_others0{i+1}' for i in range(n_bins)]
    else:
        cols = [
            f'{behavior_type}_others_never', 
            f'{behavior_type}_others_sometimes',
            f'{behavior_type}_others_half',
            f'{behavior_type}_others_often',
            f'{behavior_type}_others_always'
        ]
    
    # Agent's self-reported behavior
    self_idx = int(row[f'{behavior_type}_self']) - 1
    
    # Fill contact matrix row for this agent
    for j, col in enumerate(cols):
        if pd.notna(row[col]):
            contact_matrix[self_idx, j] = row[col]
    
    return contact_matrix

def L2 (M0, M1):
    a = np.array(M0)
    b = np.array(M1)
    diff = a - b

    return (np.sum(diff * diff))
        

def synth_matrices(distribution, h_min = 0, h_max = 5, N_h = 100):
    # given the distribution of the behavior, create vector of matrix of synthetic matrices, one for each homophily value

    
    CM_H = np.zeros((N_h, 5, 5))                # vector of synthetic matrices, one for each homophily value, for fixed distribution of behavior
    h_list = np.linspace(h_min, h_max, N_h)     # list of homophily values

    for i in range(N_h):
        CM_H[i, :, :] = create_contact_matrix(5, h_list[i], distribution)

    return CM_H

def precalculate_single_agent_raw_matrices(df, behavior_type="masks"):

    df = df.dropna()
    S_A_R_M = np.zeros((len(df), 5, 5))
    for i in range(len(df)):
        S_A_R_M[i,:,:] = generate_raw_matrix_single_agent(df, i, behavior_type = behavior_type)
    return S_A_R_M

def bootstrap_homophily(df, behavior_type, n_bootstrap=1000, seed=42):
    # Get clean dataset with no NAs in the target behavior
    df_clean = df.dropna(subset=[f'{behavior_type}_self'])
    n_samples = len(df_clean)

    # Set seed for reproducibility
    np.random.seed(seed)

    # Precalculate raw matrices
    S_A_R_M = precalculate_single_agent_raw_matrices(df, behavior_type=behavior_type)
    
    # Arrays to store bootstrap results
    bootstrap_homophily = np.zeros(n_bootstrap)
    
    for i in tqdm(range(n_bootstrap), desc=f"Bootstrapping {behavior_type}"):
        # Sample with replacement
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)

        # Sum the raw matrices for the bootstrapped sample
        raw_matrix = np.sum(S_A_R_M[bootstrap_indices, :, :], axis=0)

        # Obtain the behavior distribution of the selected bootstrap sample
        behavior_distribution = extract_behavior_distribution(df_clean.iloc[bootstrap_indices], behavior_type)
    
        # normalize the raw_matrix to the total contacts
        data_matrix = create_contact_matrix(5, None, behavior_distribution, raw_matrix)

        CM_H = synth_matrices(behavior_distribution, h_min=0, h_max=5, N_h=100)
        
        h_list = np.linspace(0, 5, 100)
        DIFFs = np.zeros(100)
        for j in range(100):
            DIFFs[j] = L2(data_matrix, CM_H[j, :, :])

        bootstrap_homophily[i] = h_list[np.argmin(DIFFs[:])]

    return {
        'bootstrap_homophily': bootstrap_homophily,
        'mean_estimate': np.mean(bootstrap_homophily),
        'ci': np.percentile(bootstrap_homophily, [2.5, 97.5])
    }


