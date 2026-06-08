# File: src/utils/visualization/heatmaps.py
"""
Heatmap visualization functions for parameter sweeps
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union

from .common import (_prepare_colormaps, _get_axis_configuration, 
                    _prepare_metric_data, _apply_final_params,
                    _plot_single_metric, _plot_single_contours)

def plot_compartment_heatmap(
    results: Dict[str, Any],
    compartment: str = "R",
    cmap: str = "viridis",
    discretize: bool = False,
    n_colors: int = 20,
    final_params: Optional[Dict[str, Any]] = None
) -> plt.Figure:
    """
    Create a heatmap showing compartment fractions across population groups with polarization on the x-axis
    
    Args:
        results: Dictionary returned by sweep_one_parameter
        compartment: The compartment to visualize as a fraction (e.g., "I" for infected)
        cmap: Colormap to use
        discretize: Whether to discretize the colormap
        n_colors: Number of discrete colors (only used if discretize=True)
        final_params: Dictionary with figure parameters for publication-quality output
            Supported keys: 'Lx', 'Ly', 'xticks', 'yticks', 'xlim', 'ylim', 'vmin', 'vmax'
        
    Returns:
        Matplotlib figure object
    """
    # Extract data
    pol_values = results['parameter_values']
    compartment_data = results['final_state'][compartment]
    n_compartments = compartment_data.shape[1]
    
    # Calculate total population for each compartment
    total_population = np.zeros_like(compartment_data)
    for comp_name, comp_data in results['final_state'].items():
        total_population += comp_data
    
    # Calculate fraction of the requested compartment
    fraction_data = np.zeros_like(compartment_data)
    mask = total_population > 0
    np.divide(compartment_data, total_population, out=fraction_data, where=mask)
    
    # For plotting (polarization, compartments)
    plot_data = fraction_data.T
    
    # Set up colormap
    if discretize:
        from .core import discretize_cmaps
        plot_cmap = discretize_cmaps(cmap, n_colors)
    else:
        plot_cmap = cmap
    
    # Get figure dimensions
    from .core import Lx, Ly
    fig_width = final_params.get('Lx', Lx)
    fig_height = final_params.get('Ly', Ly)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Get color scale limits
    vmin = final_params.get('vmin', None)
    vmax = final_params.get('vmax', None)
    
    # If vmin/vmax are not specified, determine from data
    if vmin is None:
        vmin = np.min(plot_data)
    if vmax is None:
        vmax = np.max(plot_data)
        
    # Ensure vmin and vmax are different
    if vmin == vmax:
        vmin = 0
        if vmax == 0:
            vmax = 1
    
    # Create heatmap
    im = ax.pcolormesh(
        pol_values, 
        np.arange(n_compartments), 
        plot_data, 
        cmap=plot_cmap, 
        vmin=vmin, 
        vmax=vmax
    )
    
    # Set axis limits if provided
    if 'xlim' in final_params:
        ax.set_xlim(final_params['xlim'])
    if 'ylim' in final_params:
        ax.set_ylim(final_params['ylim'])
        
    # Set custom tick positions if specified
    if 'xticks' in final_params:
        ax.set_xticks(final_params['xticks'])
    if 'yticks' in final_params:
        ax.set_yticks(final_params['yticks'])
        
    # Remove tick labels for publication plots
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Remove title and grid for publication figures
    ax.set_title('')
    ax.grid(False)
    
    return fig
def plot_compartment_heatmap2(
    results: Dict[str, Any],
    compartment: str = "R",
    cmap: str = "viridis",
    discretize: bool = False,
    n_colors: int = 20,
    fig_size: Tuple[int, int] = (10, 8),
    x_label: str = "Polarization",
    y_label: str = "Population Compartment",
    title: Optional[str] = None,
    final_params: Optional[Dict[str, Any]] = None,
    contour_values: Optional[List[float]] = None,
    contour_colors: Optional[Union[str, List[str]]] = "white",
    show_contour_labels: bool = True,
    contour_label_fmt: str = '%g',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a heatmap showing compartment fractions across population groups with polarization on the x-axis
    
    Args:
        results: Dictionary returned by sweep_one_parameter or sweep_two_parameters
        compartment: The compartment to visualize as a fraction (e.g., "I" for infected)
        cmap: Colormap to use
        discretize: Whether to discretize the colormap
        n_colors: Number of discrete colors (only used if discretize=True)
        fig_size: Figure size (width, height) - ignored if final_params is provided
        x_label: Label for x-axis
        y_label: Label for y-axis 
        title: Plot title (None for auto-generated)
        final_params: Dictionary with figure parameters for publication-quality output
            Supported keys: 'Lx', 'Ly' (figure dimensions), 
                           'xticks', 'yticks' (tick positions),
                           'xlim', 'ylim' (axis limits),
                           'vmin', 'vmax' (colorbar limits)
        contour_values: List of values at which to draw contour lines
        contour_colors: Colors for contour lines (string or list of strings)
        show_contour_labels: Whether to show labels on contour lines (default: False)
        contour_label_fmt: Format string for contour labels (default: '%g')
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure object
    """
    # Check if the requested compartment exists
    if compartment not in results['final_state']:
        raise ValueError(f"Compartment '{compartment}' not found in results")
    
    # Check if one-parameter or two-parameter sweep results
    is_one_param = 'parameter_values' in results
    is_two_param = 'parameter_grid' in results
    
    if not (is_one_param or is_two_param):
        raise ValueError("Results format not recognized - needs parameter_values or parameter_grid")
    
    # Find polarization parameter for two-parameter sweep
    if is_two_param:
        param1_name = results['parameter_names']['param1']
        param2_name = results['parameter_names']['param2']
        
        if param1_name == "beta_params":
            pol_param_idx = 0
            other_param_idx = 1
            param_name = param2_name
        elif param2_name == "beta_params":
            pol_param_idx = 1
            other_param_idx = 0
            param_name = param1_name
        else:
            # No polarization parameter found
            raise ValueError("No polarization parameter (beta_params) found in results")
            
        # Extract parameter ranges for two-parameter sweep    
        param_ranges = results['parameter_ranges']
        pol_range = param_ranges[f'param{pol_param_idx+1}']
        other_range = param_ranges[f'param{other_param_idx+1}']
        
        # Create arrays for plotting (two-parameter sweep)
        n_pol = pol_range['n']
        n_other = other_range['n']
        n_compartments = results['final_state'][compartment].shape[2]
        
        # For two-parameter sweep, we need to choose one value of the other parameter
        # Default to the middle value
        middle_idx = n_other // 2
        other_value = np.linspace(other_range['m'], other_range['M'], n_other)[middle_idx]
    else:
        # One-parameter sweep (assume it's polarization)
        n_compartments = results['final_state'][compartment].shape[1]
        param_name = "polarization"  # Default name for one-parameter sweep
    
    # Extract data based on sweep type
    if is_one_param:
        # One-parameter sweep is simpler
        pol_values = results['parameter_values']
        compartment_data = results['final_state'][compartment]
        
        # Calculate total population for each compartment
        total_population = np.zeros_like(compartment_data)
        for comp_name, comp_data in results['final_state'].items():
            total_population += comp_data
        
        # Calculate fraction of the requested compartment
        # Avoid division by zero
        fraction_data = np.zeros_like(compartment_data)
        
        # Use total population as denominator
        denominator = total_population
            
        mask = denominator > 0
        np.divide(compartment_data, denominator, out=fraction_data, where=mask)
        
        # For plotting (polarization, compartments)
        plot_data = fraction_data.T
        
    else:
        # For two-parameter sweep, extract data based on which parameter is polarization
        compartment_data = results['final_state'][compartment]
        
        # Calculate total population for each compartment
        total_population = np.zeros_like(compartment_data)
        for comp_name, comp_data in results['final_state'].items():
            total_population += comp_data
        
        # Calculate fraction of the requested compartment
        # Avoid division by zero
        fraction_data = np.zeros_like(compartment_data)
        
        # Use total population as denominator
        denominator = total_population
        
        mask = denominator > 0
        np.divide(compartment_data, denominator, out=fraction_data, where=mask)
        
        if pol_param_idx == 0:
            # Polarization is param1, reshape to (n_pol, n_other, n_compartments)
            fraction_data = fraction_data.reshape(n_pol, n_other, n_compartments)
            # For plotting, use the middle value of the other parameter
            plot_data = fraction_data[:, middle_idx, :].T
            pol_values = np.linspace(pol_range['m'], pol_range['M'], n_pol)
        else:
            # Polarization is param2, reshape to (n_other, n_pol, n_compartments)
            fraction_data = fraction_data.reshape(n_other, n_pol, n_compartments)
            # For plotting, use the middle value of the other parameter
            plot_data = fraction_data[middle_idx, :, :].T
            pol_values = np.linspace(pol_range['m'], pol_range['M'], n_pol)
    
    # Set up colormap
    if discretize:
        from .core import discretize_cmaps
        plot_cmap = discretize_cmaps(cmap, n_colors)
    else:
        plot_cmap = cmap
    
    # Check if we're creating a publication-quality plot
    is_final_plot = final_params is not None
    
    # Get figure dimensions
    if is_final_plot:
        from .core import Lx, Ly
        fig_width = final_params.get('Lx', Lx)
        fig_height = final_params.get('Ly', Ly)
    else:
        fig_width, fig_height = fig_size
    
    # Create figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Get color scale limits
    vmin = final_params.get('vmin', 0) if is_final_plot else 0
    vmax = final_params.get('vmax', 1) if is_final_plot else 1
    
    # Create heatmap
    im = ax.pcolormesh(
        pol_values, 
        np.arange(n_compartments), 
        plot_data, 
        cmap=plot_cmap, 
        vmin=vmin, 
        vmax=vmax
    )
    
    # Add contours if specified
    if contour_values is not None and len(contour_values) > 0:
        # Create coordinate grids that match the data dimensions
        X, Y = np.meshgrid(pol_values, np.arange(n_compartments))
        
        # Draw contour lines
        contour = ax.contour(
            X, Y, plot_data, 
            levels=contour_values,
            colors=contour_colors,
            linewidths=1.5,
            alpha=0.8
        )
        
        # Add contour labels based on show_contour_labels parameter
        if show_contour_labels:
            ax.clabel(contour, inline=True, fontsize=8, fmt=contour_label_fmt)
    
    # Add colorbar (only for non-final plots)
    if not is_final_plot:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f"Fraction of {compartment} relative to Population")
    
    # Set axis limits if provided in final_params
    if is_final_plot:
        if 'xlim' in final_params:
            ax.set_xlim(final_params['xlim'])
        if 'ylim' in final_params:
            ax.set_ylim(final_params['ylim'])
            
        # Set custom tick positions if specified
        if 'xticks' in final_params:
            ax.set_xticks(final_params['xticks'])
        if 'yticks' in final_params:
            ax.set_yticks(final_params['yticks'])
            
        # Remove tick labels for publication plots
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Remove title for publication figures
        ax.set_title('')
        
        # Remove grid for publication figures
        ax.grid(False)
    else:
        # Set labels and title for regular plots
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
        # Set title
        if title is None:
            if is_two_param:
                title_ref = f", {param_name}={other_value:.2f}"
            else:
                title_ref = ""
                
            title = f"Fraction of {compartment} relative to Population{title_ref}"
        
        ax.set_title(title)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure if path is provided
    fig.patch.set_visible(False)  # Make figure background transparent
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_final(
    results: Dict[str, Any],
    metric: str = "infections",
    cmap: str = "viridis",
    final_params: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a publication-quality plot for a single metric
    
    Args:
        results: Dictionary returned by sweep_two_parameters
        metric: Metric to plot ("infections", "r0", or a compartment name)
        cmap: Colormap to use
        final_params: Dictionary with figure parameters for publication-quality output
            Supported keys: 'Lx', 'Ly' (figure dimensions), 
                           'xticks', 'yticks' (tick positions),
                           'xlim', 'ylim' (axis limits),
                           'vmin', 'vmax' (colorbar limits)
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure object
    """
    # Set up default final_params if not provided
    if final_params is None:
        from .core import Lx, Ly
        final_params = {'Lx': Lx, 'Ly': Ly}
    
    # Create figure with specified dimensions
    fig_size = (final_params.get('Lx', 2.29), final_params.get('Ly', 2.16))
    
    # Use plot_multiple_metrics with a single metric and final_params
    fig = plot_multiple_metrics(
        results=results,
        metrics=[metric],
        cmaps=[cmap],
        fig_size=fig_size,
        save_path=save_path,
        final_params=final_params
    )
    
    return fig

def plot_sweep_results(
    results: Dict[str, Any], 
    metric: str = "infections",
    cmap: str = "viridis",
    fig_size: Tuple[int, int] = (12, 10),
    title_prefix: str = "",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot parameter sweep results as a heatmap
    
    Args:
        results: Dictionary returned by sweep_two_parameters
        metric: Metric to plot ("infections", "r0", or a compartment name)
        cmap: Colormap to use
        fig_size: Figure size (width, height)
        title_prefix: Prefix to add to the plot title
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure object
    """
    # Extract parameter information
    param1_name = results['parameter_names']['param1']
    param2_name = results['parameter_names']['param2']
    
    # Determine if we need to swap axes to put polarization on x-axis
    swap_axes = False
    
    # Handle the special case of beta_params and ensure polarization is on x-axis
    if param1_name == "beta_params" and param2_name != "beta_params":
        param1_label = "polarization"
        param2_label = param2_name
        swap_axes = False
    elif param2_name == "beta_params" and param1_name != "beta_params":
        param1_label = param1_name
        param2_label = "polarization"
        swap_axes = True
    else:
        param1_label = param1_name
        param2_label = param2_name
    
    model_name = results['model_name']
    
    # Get pre-shaped parameter values
    param1_vals = results['parameter_grid']['param1_vals']
    param2_vals = results['parameter_grid']['param2_vals']
    
    # Prepare data based on the requested metric
    if metric == "infections":
        # Total infections = 1 - final susceptible - final vaccinated
        data = np.ones_like(results['r0'])
        
        # Subtract final susceptible population
        if "S" in results['final_state']:
            data -= np.array(jnp.sum(results['final_state']["S"], axis=2))
            
        # Subtract final vaccinated population if it exists
        if "V" in results['final_state']:
            data -= np.array(jnp.sum(results['final_state']["V"], axis=2))
            
        title = f"{title_prefix}Total Infections"
        cbar_label = "Fraction of Population Infected"
        vmin, vmax = 0, 1  # Set fixed scale for infection metrics
        
    elif metric == "r0":
        data = np.array(results['r0'])
        title = f"{title_prefix}Basic Reproduction Number (R0)"
        cbar_label = "R0 Value"
        vmin, vmax = None, None  # Use data range for R0
        
    elif metric in results['final_state']:
        # Plot a specific compartment
        data = np.array(jnp.sum(results['final_state'][metric], axis=2))
        title = f"{title_prefix}Final {metric} Compartment"
        cbar_label = f"Fraction in {metric}"
        vmin, vmax = 0, 1  # Set fixed scale for compartment metrics
        
    else:
        raise ValueError(f"Unknown metric: {metric}. Options are 'infections', 'r0', or a compartment name.")
    
    # If swapping axes, transpose data and parameter arrays
    if swap_axes:
        data = data.T
        param1_vals, param2_vals = param2_vals.T, param1_vals.T
        param1_label, param2_label = param2_label, param1_label
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Create heatmap with fixed scale for applicable metrics
    im = ax.pcolormesh(param1_vals, param2_vals, data, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    
    # Set labels and title
    ax.set_xlabel(param1_label)
    ax.set_ylabel(param2_label)
    ax.set_title(f"{model_name} Model: {title}")
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_multiple_metrics(
    results: Dict[str, Any],
    metrics: List[str] = ["infections", "r0"],
    cmaps: Optional[List[str]] = None,
    fig_size: Tuple[int, int] = (18, 8),
    save_path: Optional[str] = None,
    i_threshold: float = 1e-4,
    contour_values: Optional[List[List[float]]] = None,
    contour_colors: Optional[List[List[str]]] = None,
    final_params: Optional[Dict[str, Any]] = None,
    rect_coords: Optional[List[float]] = None  # [x0, y0, width, height]
) -> plt.Figure:
    """
    Plot multiple metrics from parameter sweep results
    
    Args:
        results: Dictionary returned by sweep_two_parameters
        metrics: List of metrics to plot ("infections", "r0", "I_threshold", or compartment names)
        cmaps: List of colormaps to use (should match length of metrics)
        fig_size: Figure size (width, height)
        save_path: Path to save the figure (if None, figure is not saved)
        i_threshold: Threshold for I compartment fraction (used for "I_threshold" metric)
        contour_values: List of lists with contour values for each metric (None or empty list means no contours)
        contour_colors: List of lists with contour colors for each metric (None means default white color)
                        Example: [['#e5f5f9','#99d8c9','#2ca25f'], None, None, None, ['red', 'orange', 'yellow', 'green']]
        final_params: Optional dictionary with figure parameters for publication-quality output (only used for single metric)
                      Supported keys: 'Lx', 'Ly' (figure dimensions), 'xticks', 'yticks', 'xlim', 'ylim'
        rect_coords: Optional coordinates for rectangle overlay [x0, y0, width, height]
        
    Returns:
        Matplotlib figure object
    """
    n_plots = len(metrics)
    
    # Determine if this is a publication-ready plot
    is_final_plot = n_plots == 1 and final_params is not None
    
    # Set default colormaps if not provided
    cmaps = _prepare_colormaps(cmaps, n_plots)
    
    # Prepare contour values (None for metrics without contours)
    if contour_values is None:
        contour_values = [None] * n_plots
    elif len(contour_values) < n_plots:
        contour_values.extend([None] * (n_plots - len(contour_values)))
    
    # Prepare contour colors (None for default colors)
    if contour_colors is None:
        contour_colors = [None] * n_plots
    elif len(contour_colors) < n_plots:
        contour_colors.extend([None] * (n_plots - len(contour_colors)))
    
    # Use final_params for figure size if specified and this is a single-metric plot
    if is_final_plot and 'Lx' in final_params and 'Ly' in final_params:
        custom_fig_size = (final_params['Lx'], final_params['Ly'])
    else:
        custom_fig_size = fig_size
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_plots, figsize=custom_fig_size)
    if n_plots == 1:
        axes = [axes]  # Make axes iterable if there's only one subplot
    
    # Extract axis information and determine configuration
    axis_config = _get_axis_configuration(results)
    
    model_name = results['model_name']
    
    # Plot each metric
    for i, (metric, cmap, contours, colors) in enumerate(zip(metrics, cmaps, contour_values, contour_colors)):
        ax = axes[i]
        
        # Get data, title, and visualization parameters for the metric
        metric_data = _prepare_metric_data(results, metric, i_threshold)
        
        if metric_data is None:
            continue  # Skip unknown metrics
        
        # Plot the data with or without colorbar
        _plot_single_metric(fig, ax, metric_data, axis_config, cmap, show_colorbar=not is_final_plot)
        
        # Add contours if specified (skip if None or empty list)
        if contours and len(contours) > 0:
            _plot_single_contours(ax, metric_data, axis_config, contours, colors)
            
        # Add rectangle overlay if coordinates are provided
        if rect_coords is not None:
            x0, y0, width, height = rect_coords
            rect = plt.Rectangle((x0, y0), width, height, fill=False, 
                                  edgecolor='black', linewidth=2, alpha=1)
            ax.add_patch(rect)
    
    # Apply final parameters if this is a single-metric plot
    if is_final_plot:
        _apply_final_params(fig, axes[0], final_params)
    elif final_params is not None:
        print("Warning: final_params is only applied to single-metric plots. Ignoring for multi-metric plot.")
    
    # Set overall title if not in final params mode
    if not is_final_plot:
        fig.suptitle(f"{model_name} Model: Parameter Sweep Results", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_compartment_fractions(
    results: Dict[str, Any],
    compartment: str = "I",  # The compartment to show as a fraction (e.g., "I" for infected)
    cmap: str = "viridis",
    fig_size: Tuple[int, int] = (12, 10),
    title_prefix: str = "",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a heatmap showing the fraction of a specific compartment (e.g., infected)
    across population groups, with polarization on the x-axis.
    
    Args:
        results: Dictionary returned by sweep_two_parameters
        compartment: The compartment to display as a fraction (e.g., "I" for infected)
        cmap: Colormap to use
        fig_size: Figure size (width, height)
        title_prefix: Prefix to add to the plot title
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure object
    """
    # Check if the requested compartment exists
    if compartment not in results['final_state']:
        raise ValueError(f"Compartment '{compartment}' not found in results")
    
    # Extract parameter information
    param1_name = results['parameter_names']['param1']
    param2_name = results['parameter_names']['param2']
    
    # Find the polarization parameter
    if param1_name == "beta_params":
        pol_param_idx = 0
        other_param_idx = 1
        x_label = "Polarization"
        y_label = param2_name
    elif param2_name == "beta_params":
        pol_param_idx = 1
        other_param_idx = 0
        x_label = "Polarization"
        y_label = param1_name
    else:
        # No polarization parameter found
        raise ValueError("No polarization parameter (beta_params) found in results")
    
    model_name = results['model_name']
    
    # Get parameter ranges
    param_ranges = results['parameter_ranges']
    pol_range = param_ranges[f'param{pol_param_idx+1}']
    other_range = param_ranges[f'param{other_param_idx+1}']
    
    # Create arrays for plotting
    n_pol = pol_range['n']
    n_other = other_range['n']
    n_compartments = results['final_state'][compartment].shape[2]
    
    # Extract the compartment data
    compartment_data = results['final_state'][compartment]
    
    # Calculate total population for each compartment
    total_population = np.zeros_like(compartment_data)
    for comp_name, comp_data in results['final_state'].items():
        total_population += comp_data
    
    # Calculate fraction of the requested compartment
    # Avoid division by zero
    fraction_data = np.zeros_like(compartment_data)
    mask = total_population > 0
    np.divide(compartment_data, total_population, out=fraction_data, where=mask)
    
    # Prepare data based on which parameter is polarization
    if pol_param_idx == 0:
        # Polarization is param1, reshape to (n_pol, n_other, n_compartments)
        fraction_data = fraction_data.reshape(n_pol, n_other, n_compartments)
        # For plotting, we need (n_pol, n_compartments)
        # Let's use the middle value of the other parameter
        middle_idx = n_other // 2
        plot_data = fraction_data[:, middle_idx, :]
        pol_values = np.linspace(pol_range['m'], pol_range['M'], n_pol)
        other_value = np.linspace(other_range['m'], other_range['M'], n_other)[middle_idx]
    else:
        # Polarization is param2, reshape to (n_other, n_pol, n_compartments)
        fraction_data = fraction_data.reshape(n_other, n_pol, n_compartments)
        # For plotting, we need (n_pol, n_compartments)
        # Let's use the middle value of the other parameter
        middle_idx = n_other // 2
        plot_data = fraction_data[middle_idx, :, :].T  # Transpose to get (n_compartments, n_pol)
        pol_values = np.linspace(pol_range['m'], pol_range['M'], n_pol)
        other_value = np.linspace(other_range['m'], other_range['M'], n_other)[middle_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Create heatmap
    im = ax.pcolormesh(pol_values, np.arange(n_compartments), plot_data, cmap=cmap, vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Fraction in {compartment}")
    
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel("Population Compartment")
    fixed_param_str = f", {y_label}={other_value:.2f}"
    ax.set_title(f"{model_name} Model: {title_prefix}Fraction of {compartment} by Population Group{fixed_param_str}")
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig