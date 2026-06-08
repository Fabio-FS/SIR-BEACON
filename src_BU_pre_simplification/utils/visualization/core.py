# File: src/utils/visualization/core.py
"""
Core visualization utilities for epidemic models
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from matplotlib.colors import ListedColormap

# Standard figure dimensions for publication-quality plots
Lx = 2.29
Ly = 2.16

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

# Create custom colormaps

# my_hot_r is used for prevalence heatmaps
my_hot_r = discretize_cmaps('hot_r', 12)
my_hot_r.set_bad('gray')

# my_vir_r is used for R0 heatmaps
my_vir_r = discretize_cmaps('viridis_r', 12)
my_vir_r.set_bad('gray')


CH = ["#00441b", "#238b45", "#000", "#66c2a4", "#99d8c9"]       # used for fixed homophily
CP = ["#000", "#7f0000", "#d7301f", "#fc8d59", "#fdbb84"]       # used for fixed polarization



def print_sweep_results(results: Dict[str, Any], verbose: bool = False) -> None:
    """
    Print sweep results in a readable format
    
    Args:
        results: Dictionary returned by sweep_two_parameters
        verbose: Whether to print detailed state information
    """
    print(f"Model: {results['model_name']}")
    print("\nParameters swept:")
    
    # Handle the special case of beta_params
    param1_name = results['parameter_names']['param1']
    param1_label = "polarization" if param1_name == "beta_params" else param1_name
    
    param2_name = results['parameter_names']['param2']
    param2_label = "polarization" if param2_name == "beta_params" else param2_name
    
    print(f"  {param1_label}: {results['parameter_ranges']['param1']['m']} to {results['parameter_ranges']['param1']['M']} ({results['parameter_ranges']['param1']['n']} points)")
    print(f"  {param2_label}: {results['parameter_ranges']['param2']['m']} to {results['parameter_ranges']['param2']['M']} ({results['parameter_ranges']['param2']['n']} points)")
    
    print("\nResults summary:")
    print(f"  Total parameter combinations: {len(results['r0'])}")
    print(f"  R0 range: {jnp.nanmin(results['r0']):.4f} to {jnp.nanmax(results['r0']):.4f}")
    print(f"  Homophily measure: {jnp.nanmean(results['homophily']):.4f}")
    
    # Print compartment states summary
    print("\nFinal state compartments:")
    for compartment, values in results['final_state'].items():
        avg_value = jnp.nanmean(jnp.sum(values, axis=2))
        min_value = jnp.nanmin(jnp.sum(values, axis=2))
        max_value = jnp.nanmax(jnp.sum(values, axis=2))
        print(f"  {compartment}: avg={avg_value:.4f}, min={min_value:.4f}, max={max_value:.4f}")
    
    # Print detailed information if requested
    if verbose:
        print("\nParameter grid (first 5 combinations):")
        for i in range(min(5, len(results['parameter_grid']))):
            val1 = results['parameter_grid'][i][0]
            val2 = results['parameter_grid'][i][1]
            
            # Format parameter values
            val1_str = f"{val1:.4f}"
            val2_str = f"{val2:.4f}"
            
            print(f"  {i+1}: {param1_label}={val1_str}, {param2_label}={val2_str}")
            print(f"     R0: {results['r0'][i]:.4f}")
            for compartment, values in results['final_state'].items():
                print(f"     {compartment} total: {jnp.sum(values[i]):.4f}")
        
        if len(results['parameter_grid']) > 5:
            print("  ...")





def run_pol_hom_effect_analysis(
    model_module,
    discrete_param_name,
    discrete_param_values,
    continuous_param_name="beta_params",
    continuous_param_range={"m": 0, "M": 1, "n": 50},
    compartment_to_analyze="R",
    custom_base_params=None,
    publication_ready=False,
    colors=None,
    xticks=None,
    yticks=None,
    xticklabels=None,
    yticklabels=None,
    show_legend=False,
    custom_ylim=None,
    custom_xlim=None,
    save_path=None
):
    """
    Plot percentage change from baseline (consensus, no homophily) across parameter values.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from src.utils.batch_sweep import sweep_two_parameters
    
    # Set up default parameters
    default_colors = ["#000", "#7f0000", "#d7301f", "#fc8d59", "#fdbb84"]
    fig_dimensions = (2.29, 2.16) if publication_ready else (10, 6)
    
    # Format discrete parameter values
    if not isinstance(discrete_param_values, dict):
        param_range = {"m": min(discrete_param_values), "M": max(discrete_param_values), "n": len(discrete_param_values)}
        discrete_values_list = discrete_param_values
    else:
        param_range = discrete_param_values
        discrete_values_list = np.linspace(param_range["m"], param_range["M"], param_range["n"])
    
    # Format continuous parameter range
    if isinstance(continuous_param_range, list):
        continuous_param_dict = {"m": min(continuous_param_range), "M": max(continuous_param_range), 
                               "n": len(continuous_param_range)}
    else:
        continuous_param_dict = continuous_param_range
        
    # Run the full parameter sweep
    results_full = sweep_two_parameters(
        model_module=model_module,
        param1_name=continuous_param_name,
        param1_range=continuous_param_dict,
        param2_name=discrete_param_name,
        param2_range=param_range,
        custom_base_params=custom_base_params or model_module.get_default_params()
    )
    
    # Extract data and x-values
    compartment_data = np.array(np.sum(results_full['final_state'][compartment_to_analyze], axis=2))
    x_values = np.linspace(continuous_param_dict["m"], continuous_param_dict["M"], continuous_param_dict["n"])
    
    # Run baseline (minimal polarization, zero homophily)
    results_baseline = sweep_two_parameters(
        model_module=model_module,
        param1_name="beta_params",
        param1_range={"m": 0.00001, "M": 0.00001, "n": 1},
        param2_name="homophilic_tendency",
        param2_range={"m": 0, "M": 0, "n": 1},
        custom_base_params=custom_base_params or model_module.get_default_params()
    )
    
    baseline_data = np.array(np.sum(results_baseline['final_state'][compartment_to_analyze], axis=2))
    
    # Create plot with transparent background
    fig, ax = plt.subplots(figsize=fig_dimensions)
    fig.patch.set_visible(False)  # Make figure background transparent
    
    # Plot percentage differences from baseline
    for i, val in enumerate(discrete_values_list):
        percentage_diff = (compartment_data[i] - baseline_data[0]) / baseline_data[0] * 100
        color = colors[i] if colors and i < len(colors) else default_colors[i % len(default_colors)]
        ax.plot(x_values, percentage_diff, color=color, label=f"{discrete_param_name}={val}")
    
    # Apply styling
    if publication_ready:
        # Publication styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        
        # Apply custom settings if provided
        if xticks is not None: ax.set_xticks(xticks)
        if yticks is not None: ax.set_yticks(yticks)
        if xticklabels is not None: ax.set_xticklabels(xticklabels)
        if yticklabels is not None: ax.set_yticklabels(yticklabels)
    else:
        # Standard styling
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel(continuous_param_name)
        ax.set_ylabel(f"% Change in {compartment_to_analyze}")
    ax.plot([min(x_values), max(x_values)], [0, 0], 'k--')
    
    # Set axis limits if specified
    if custom_xlim is not None: ax.set_xlim(custom_xlim)
    if custom_ylim is not None: ax.set_ylim(custom_ylim)
    
    # Add legend if requested
    if show_legend: ax.legend(loc="best", frameon=False)
    
    # Save if requested
    
    fig.patch.set_visible(False)  # Make figure background transparent
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    
    return fig, results_full, results_baseline


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
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    
    return fig