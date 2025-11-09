# File: src/utils/visualization/common.py
"""
Common utilities for visualization functions
"""
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

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
