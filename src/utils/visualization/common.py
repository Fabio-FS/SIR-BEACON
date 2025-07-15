# File: src/utils/visualization/common.py
"""
Common utilities for visualization functions
"""
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt

def _prepare_colormaps(cmaps: Optional[List[str]], n_plots: int) -> List[str]:
    """Prepare colormap list for plots"""
    if cmaps is None:
        default_cmaps = ["viridis", "plasma", "inferno", "magma", "cividis"]
        # Cycle through the list if there are more metrics than colormaps
        return [default_cmaps[i % len(default_cmaps)] for i in range(n_plots)]
    return cmaps

def _get_axis_configuration(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and process axis configuration from results"""
    param1_name = results['parameter_names']['param1']
    param2_name = results['parameter_names']['param2']
    
    # Determine if we need to swap axes to put polarization on x-axis
    swap_axes = False
    flag_force_xlim = False

    # Handle the special case of beta_params and ensure polarization is on x-axis
    if param1_name == "beta_params" and param2_name != "beta_params":
        param1_label = "polarization"
        param2_label = param2_name
        swap_axes = False
        flag_force_xlim = True
    elif param2_name == "beta_params" and param1_name != "beta_params":
        param1_label = param1_name
        param2_label = "polarization"
        swap_axes = True
        flag_force_xlim = True
    else:
        param1_label = param1_name
        param2_label = param2_name
    
    # Get pre-shaped parameter values
    param1_vals = results['parameter_grid']['param1_vals']
    param2_vals = results['parameter_grid']['param2_vals']
    
    return {
        'param1_label': param1_label,
        'param2_label': param2_label,
        'param1_vals': param1_vals,
        'param2_vals': param2_vals,
        'swap_axes': swap_axes,
        'force_xlim': flag_force_xlim
    }

def _prepare_metric_data(results: Dict[str, Any], metric: str, i_threshold: float) -> Optional[Dict[str, Any]]:
    """Prepare data for a specific metric"""
    from .core import my_vir_r  # Local import to avoid circular dependency
    
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
            
        return {
            'data': data,
            'title': "Total Infections",
            'cbar_label': "Fraction Infected",
            'vmin': 0,
            'vmax': 1,
            'use_custom_cbar': False
        }
        
    elif metric == "r0":
        data = np.array(results['r0'])
        return {
            'data': data,
            'title': "Basic Reproduction Number (R0)",
            'cbar_label': "R0 Value",
            'vmin': None,
            'vmax': None,
            'use_custom_cbar': False
        }
        
    elif metric == "I_threshold":
        # Check if I compartment exceeds threshold
        if "I" in results['final_state']:
            i_fractions = np.array(jnp.sum(results['final_state']["I"], axis=2))
            data = (i_fractions > i_threshold).astype(float)
            
            # Create binary colormap with 2 colors
            binary_cmap = my_vir_r
            
            return {
                'data': data,
                'title': f"I > {i_threshold:.0e} Threshold",
                'cbar_label': "",
                'vmin': 0,
                'vmax': 1,
                'use_custom_cbar': True,
                'custom_cmap': binary_cmap,
                'is_threshold': True
            }
        return None  # Skip if I compartment doesn't exist
        
    elif metric in results['final_state']:
        # Plot a specific compartment
        data = np.array(jnp.sum(results['final_state'][metric], axis=2))
        return {
            'data': data,
            'title': f"Final {metric} Compartment",
            'cbar_label': f"Fraction in {metric}",
            'vmin': 0,
            'vmax': 1,
            'use_custom_cbar': False
        }
        
    return None  # Unknown metric

def _apply_final_params(fig, ax, params: Dict[str, Any]) -> None:
    """Apply publication-quality parameters to a figure"""
    # Make figure background transparent
    fig.patch.set_visible(False)
    
    # Set axis limits if specified
    if 'xlim' in params:
        ax.set_xlim(params['xlim'])
    if 'ylim' in params:
        ax.set_ylim(params['ylim'])
    
    # Set custom tick positions if specified
    if 'xticks' in params:
        ax.set_xticks(params['xticks'])
    if 'yticks' in params:
        ax.set_yticks(params['yticks'])
    
    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Remove title for publication figures
    ax.set_title('')
    
    # Remove grid for publication figures
    ax.grid(False)
    
    # Apply colorbar limits if specified
    if hasattr(ax, 'collections') and len(ax.collections) > 0:
        # Get the first collection (usually the pcolormesh)
        collection = ax.collections[0]
        
        # Update colorbar limits if provided
        if 'vmin' in params or 'vmax' in params:
            vmin = params.get('vmin', collection.get_clim()[0])
            vmax = params.get('vmax', collection.get_clim()[1])
            collection.set_clim(vmin, vmax)
            
            # If there's a colorbar associated with this axes, update it
            for cbar in fig.get_axes():
                if hasattr(cbar, 'ax') and isinstance(cbar.ax, plt.Axes) and cbar.ax.get_position().x0 > ax.get_position().x1:
                    # This is likely the colorbar for our axis
                    cbar.mappable.set_clim(vmin, vmax)
                    break


def _plot_single_metric(fig, ax, metric_data: Dict[str, Any], 
                       axis_config: Dict[str, Any], cmap: str, show_colorbar: bool = True) -> None:
    """Plot a single metric on the provided axis"""
    data = metric_data['data']
    
    # If swapping axes, transpose data
    if axis_config['swap_axes']:
        data = data.T
        plot_param1_vals = axis_config['param2_vals'].T
        plot_param2_vals = axis_config['param1_vals'].T
        x_label = axis_config['param2_label']
        y_label = axis_config['param1_label']
    else:
        plot_param1_vals = axis_config['param1_vals']
        plot_param2_vals = axis_config['param2_vals']
        x_label = axis_config['param1_label']
        y_label = axis_config['param2_label']
    
    # Use custom colormap if provided
    if metric_data.get('use_custom_cbar') and metric_data.get('custom_cmap') is not None:
        cmap = metric_data['custom_cmap']
    
    # Create heatmap - explicitly set rasterized=False
    im = ax.pcolormesh(
        plot_param1_vals, 
        plot_param2_vals, 
        data, 
        cmap=cmap, 
        vmin=metric_data['vmin'], 
        vmax=metric_data['vmax'],
        rasterized=False  # This is the key change - ensure vector output
    )
    
    # Add colorbar if requested
    if show_colorbar:
        if metric_data.get('use_custom_cbar') and metric_data.get('is_threshold'):
            cbar = fig.colorbar(im, ax=ax, ticks=[0.25, 0.75])
            cbar.ax.set_yticklabels(["Below\nthreshold", "Above\nthreshold"])
        else:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(metric_data['cbar_label'])
    
    # Set labels and title (only if not in final mode)
    if show_colorbar:
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(metric_data['title'])
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)


def _plot_single_contours(ax, metric_data: Dict[str, Any], 
                         axis_config: Dict[str, Any], contour_levels: List[float],
                         colors: Optional[List[str]] = None) -> None:
    """Add contour lines to an existing plot"""
    data = metric_data['data']
    
    # If swapping axes, transpose data
    if axis_config['swap_axes']:
        data = data.T
        plot_param1_vals = axis_config['param2_vals'].T
        plot_param2_vals = axis_config['param1_vals'].T
    else:
        plot_param1_vals = axis_config['param1_vals']
        plot_param2_vals = axis_config['param2_vals']
    
    # Get data dimensions
    ny, nx = data.shape
    
    # Create coordinate grids that match the data dimensions
    x_contour = np.linspace(plot_param1_vals.min(), plot_param1_vals.max(), nx)
    y_contour = np.linspace(plot_param2_vals.min(), plot_param2_vals.max(), ny)
    X, Y = np.meshgrid(x_contour, y_contour)
    
    # Determine colors for contour lines
    if colors is None:
        contour_colors = 'white'  # Default color
    else:
        # If single color provided for all contours
        if len(colors) == 1:
            contour_colors = colors[0]
        # If multiple colors provided (one per level)
        elif len(colors) >= len(contour_levels):
            contour_colors = colors[:len(contour_levels)]
        # If fewer colors than levels, cycle through the colors
        else:
            contour_colors = []
            for i in range(len(contour_levels)):
                contour_colors.append(colors[i % len(colors)])
    
    # Draw contour lines - explicitly set rasterized=False
    contour = ax.contour(
        X, Y, data, 
        levels=contour_levels,
        colors=contour_colors,
        linewidths=1.5,
        alpha=0.8,
        rasterized=False  # Ensure vector output for contours
    )
    
    # Add contour labels
    ax.clabel(contour, inline=True, fontsize=8, fmt='%g')



def _get_centers(arr: np.ndarray) -> np.ndarray:
    """Get cell centers from cell edges for contour plotting"""
    # For 2D arrays, extract the first row/column
    if arr.ndim > 1:
        if arr.shape[0] > 1:
            arr = arr[0, :]  # Get first row if multiple rows
        else:
            arr = arr[:, 0]  # Get first column if only one row
            
    # Calculate centers between adjacent values
    centers = (arr[:-1] + arr[1:]) / 2
    
    # Add extrapolated points at the ends to match the size of the original mesh
    first_center = 2 * centers[0] - centers[1]
    last_center = 2 * centers[-1] - centers[-2]
    
    return np.concatenate([[first_center], centers, [last_center]])

def plot_polarization_vs_percent_change_publication(
    isolated_results,
    homophily_results_list,
    compartment="R",
    param_labels=None,
    colors=None,
    x_label="",
    y_label="",
    fig_size=(2.29, 2.16),  # Standard publication size from your heatmap code
    xticks=None,
    yticks=None,
    xticklabels=None,
    yticklabels=None,
    xlim=None,
    ylim=None,
    save_path=None,
    show_legend=False,
    legend_loc="best",
    legend_frameon=False,
    linestyles=None,
    linewidth=1.5
):
    """
    Create a publication-ready plot showing percent change in infections vs polarization
    
    Args:
        isolated_results: Results from isolated model (baseline)
        homophily_results_list: List of results with different homophily settings
        compartment: Compartment to analyze (usually "R" or "I")
        param_labels: List of labels for each homophily setting
        colors: List of colors for each homophily setting
        x_label, y_label: Axis labels (empty for publication style)
        fig_size: Figure size in inches (publication dimensions)
        xticks, yticks: Custom tick positions
        xticklabels, yticklabels: Custom tick labels (None to hide labels)
        xlim, ylim: Axis limits
        save_path: Path to save the figure
        show_legend: Whether to show the legend
        legend_loc: Legend location
        legend_frameon: Whether to show a frame around the legend
        linestyles: List of line styles for each model
        linewidth: Width of the lines
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import jax.numpy as jnp
    
    # Setup default colors if not provided
    if colors is None:
        colors = ["#000", "#7f0000", "#d7301f", "#fc8d59", "#fdbb84"]
    
    # Setup default param labels if not provided
    if param_labels is None and len(homophily_results_list) > 0:
        param_labels = [f"Model {i+1}" for i in range(len(homophily_results_list))]
    
    # Setup default linestyles if not provided
    if linestyles is None:
        linestyles = ['-'] * len(homophily_results_list)
    elif len(linestyles) < len(homophily_results_list):
        # Extend linestyles if not enough provided
        linestyles = linestyles + ['-'] * (len(homophily_results_list) - len(linestyles))
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    fig.patch.set_visible(False)  # Make figure background transparent
    
    # Get the polarization values (assume same for all models)
    polarization_values = isolated_results['parameter_values']
    
    # Extract baseline data for the selected compartment
    if compartment in isolated_results['final_state']:
        # Sum across population groups
        baseline_data = np.array(jnp.sum(isolated_results['final_state'][compartment], axis=1))
    else:
        raise ValueError(f"Compartment {compartment} not found in isolated_results")
    
    # Plot for each homophily model
    for i, results in enumerate(homophily_results_list):
        if compartment in results['final_state']:
            # Sum across population groups for this model
            model_data = np.array(jnp.sum(results['final_state'][compartment], axis=1))
            
            # Calculate percent change: ((model - baseline) / baseline) * 100
            percent_change = ((model_data - baseline_data) / baseline_data) * 100
            
            # Plot the data - publication style with no markers
            label = param_labels[i] if param_labels and i < len(param_labels) else None
            color = colors[i % len(colors)]
            linestyle = linestyles[i % len(linestyles)]
            ax.plot(polarization_values, percent_change, 
                   color=color, 
                   label=label, 
                   linestyle=linestyle,
                   linewidth=linewidth)
        else:
            print(f"Warning: Compartment {compartment} not found in results for model {i}")
    
    # Add horizontal line at 0% change as a dashed black line
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.8, linewidth=0.75)
    
    # Publication styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    
    # Add labels (typically empty for publication plots)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # Set custom ticks if provided
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    
    # Set tick labels (or hide them for publication)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    else:
        ax.set_xticklabels([])
        
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    else:
        ax.set_yticklabels([])
    
    # Set axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(min(polarization_values), max(polarization_values))
        
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Add legend if requested
    if show_legend and param_labels and len(param_labels) > 0:
        ax.legend(loc=legend_loc, frameon=legend_frameon)
    
    # Save figure if path is provided
    fig.patch.set_visible(False)  # Make figure background transparent
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    return fig

def parse_2_parameters(
    results: Dict[str, Any], 
    metrics: List[str] = ["infections", "r0"],
    i_threshold: float = 1e-4
) -> Dict[str, Any]:
    """
    Parse results from sweep_two_parameters and prepare data matrices for visualization
    
    This function separates data processing from visualization, taking the raw results
    from sweep_two_parameters and preparing clean matrices for each requested metric.
    
    Args:
        results: Dictionary returned by sweep_two_parameters
        metrics: List of metrics to process ("infections", "r0", "I_threshold", or compartment names)
        i_threshold: Threshold for I compartment fraction (used for "I_threshold" metric)
        
    Returns:
        Dictionary containing:
        - 'data': Dictionary with metrics as keys and processed data matrices as values
        - 'axis_config': Configuration for axes (labels, values, etc.)
        - 'metadata': Additional metadata for plotting (titles, labels, ranges)
    """
    # Extract axis information
    axis_config = _get_axis_configuration(results)
    
    # Initialize data dictionary
    data_dict = {}
    metadata = {}
    
    # Process each requested metric
    for metric in metrics:
        # Get data and metadata for the metric
        metric_data = _prepare_metric_data(results, metric, i_threshold)
        
        if metric_data is None:
            continue  # Skip unknown metrics
        
        # Store the processed data matrix
        data_dict[metric] = metric_data['data']
        
        # Store metadata for visualization
        metadata[metric] = {
            'title': metric_data['title'],
            'cbar_label': metric_data['cbar_label'],
            'vmin': metric_data['vmin'],
            'vmax': metric_data['vmax'],
            'use_custom_cbar': metric_data.get('use_custom_cbar', False)
        }
        
        # Add custom colormap if present
        if 'custom_cmap' in metric_data:
            metadata[metric]['custom_cmap'] = metric_data['custom_cmap']
            
        # Add threshold flag if present
        if 'is_threshold' in metric_data:
            metadata[metric]['is_threshold'] = metric_data['is_threshold']
    
    # Return the processed data and configuration
    return {
        'data': data_dict,
        'axis_config': axis_config,
        'metadata': metadata,
        'model_name': results['model_name']
    }

def simple_parse_2_parameters(
    results: Dict[str, Any], 
    metric: str = "infections",
    i_threshold: float = 1e-4
) -> np.ndarray:
    """
    Parse results from sweep_two_parameters and return a simple matrix for a single metric
    
    This function extracts a clean data matrix for the requested metric.
    
    Args:
        results: Dictionary returned by sweep_two_parameters
        metric: Metric to process ("infections", "r0", "I_threshold", or compartment names)
        i_threshold: Threshold for I compartment fraction (used for "I_threshold" metric)
        
    Returns:
        Numpy array containing the data matrix for the requested metric
    """
    # Get data for the metric
    metric_data = _prepare_metric_data(results, metric, i_threshold)
    
    if metric_data is None:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Return just the data matrix
    return metric_data['data']

def plot_metrics(
    parsed_data: Dict[str, Any],
    cmaps: Optional[List[str]] = None,
    fig_size: Tuple[int, int] = (18, 8),
    save_path: Optional[str] = None,
    contour_values: Optional[Dict[str, List[float]]] = None,
    contour_colors: Optional[Dict[str, List[str]]] = None,
    final_params: Optional[Dict[str, Any]] = None,
    rect_coords: Optional[List[float]] = None  # [x0, y0, width, height]
) -> plt.Figure:
    """
    Plot multiple metrics using pre-processed data from parse_2_parameters
    
    This function focuses solely on visualization, taking pre-processed data matrices
    and creating a multi-panel figure.
    
    Args:
        parsed_data: Dictionary returned by parse_2_parameters
        cmaps: List or dictionary of colormaps to use (should match metrics)
        fig_size: Figure size (width, height)
        save_path: Path to save the figure (if None, figure is not saved)
        contour_values: Dictionary with metrics as keys and lists of contour values as values
        contour_colors: Dictionary with metrics as keys and lists of contour colors as values
        final_params: Optional dictionary with figure parameters for publication-quality output
                      (only used for single metric)
        rect_coords: Optional coordinates for rectangle overlay [x0, y0, width, height]
        
    Returns:
        Matplotlib figure object
    """
    # Extract components from parsed data
    data_dict = parsed_data['data']
    axis_config = parsed_data['axis_config']
    metadata = parsed_data['metadata']
    model_name = parsed_data['model_name']
    
    # Get list of metrics that have data
    metrics = list(data_dict.keys())
    n_plots = len(metrics)
    
    # Determine if this is a publication-ready plot
    is_final_plot = n_plots == 1 and final_params is not None
    
    # Process colormap inputs
    if cmaps is None:
        # Default colormaps for each metric
        cmaps = _prepare_colormaps(None, n_plots)
        cmap_dict = {metric: cmaps[i] for i, metric in enumerate(metrics)}
    elif isinstance(cmaps, list):
        # List of colormaps
        cmaps = _prepare_colormaps(cmaps, n_plots)
        cmap_dict = {metric: cmaps[i] for i, metric in enumerate(metrics)}
    elif isinstance(cmaps, dict):
        # Dictionary mapping metrics to colormaps
        cmap_dict = cmaps
    else:
        raise ValueError("cmaps must be None, a list, or a dictionary")
    
    # Process contour inputs
    if contour_values is None:
        contour_values = {}
    
    if contour_colors is None:
        contour_colors = {}
    
    # Use final_params for figure size if specified and this is a single-metric plot
    if is_final_plot and 'Lx' in final_params and 'Ly' in final_params:
        custom_fig_size = (final_params['Lx'], final_params['Ly'])
    else:
        custom_fig_size = fig_size
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_plots, figsize=custom_fig_size)
    if n_plots == 1:
        axes = [axes]  # Make axes iterable if there's only one subplot
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Get data and metadata for this metric
        data = data_dict[metric]
        metric_meta = metadata[metric]
        
        # Get colormap for this metric (use custom if specified)
        if metric_meta.get('use_custom_cbar') and 'custom_cmap' in metric_meta:
            cmap = metric_meta['custom_cmap']
        else:
            cmap = cmap_dict.get(metric, 'viridis')
        
        # Plot the data with or without colorbar
        if metric_meta.get('use_custom_cbar') and metric_meta.get('is_threshold') and not is_final_plot:
            # Data has been processed as a threshold - use custom colorbar
            im = _plot_single_metric(fig, ax, {
                'data': data,
                'title': metric_meta['title'],
                'cbar_label': metric_meta['cbar_label'],
                'vmin': metric_meta['vmin'],
                'vmax': metric_meta['vmax'],
                'use_custom_cbar': True,
                'custom_cmap': cmap,
                'is_threshold': True
            }, axis_config, cmap, show_colorbar=not is_final_plot)
        else:
            # Standard plotting
            im = _plot_single_metric(fig, ax, {
                'data': data,
                'title': metric_meta['title'],
                'cbar_label': metric_meta['cbar_label'],
                'vmin': metric_meta['vmin'],
                'vmax': metric_meta['vmax']
            }, axis_config, cmap, show_colorbar=not is_final_plot)
        
        # Add contours if specified for this metric
        if metric in contour_values and contour_values[metric]:
            contours = contour_values[metric]
            colors = contour_colors.get(metric, None)
            _plot_single_contours(ax, {'data': data}, axis_config, contours, colors)
            
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
