# File: src/utils/visualization/slices.py
"""
Visualization functions for parameter slices
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union

# Import from sensitivity module instead of local core
from ..sensitivity.core import get_slice, reshape_results

def plot_parameter_slice(
    results: Dict[str, Any],
    fixed_params: Dict[str, Union[float, int]],
    output_key: str = "r0",
    title: Optional[str] = None,
    cmap: str = "viridis",
    fig_size: Tuple[int, int] = (10, 8),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    add_contours: bool = False,
    contour_levels: Optional[List[float]] = None,
    contour_colors: Optional[Union[str, List[str]]] = "white",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize a 2D slice from n-dimensional parameter sweep results
    
    Args:
        results: Results dictionary from sweep_n_parameters
        fixed_params: Dictionary mapping parameter names to fixed values
                     Must fix n-2 parameters to get a 2D slice
        output_key: The output to visualize
        title: Plot title (None for auto-generated)
        cmap: Colormap to use
        fig_size: Figure size
        vmin, vmax: Color scale limits (None for auto)
        add_contours: Whether to add contour lines
        contour_levels: Levels for contour lines (None for auto)
        contour_colors: Colors for contour lines
        save_path: Path to save figure (None to not save)
        
    Returns:
        Matplotlib figure
    """
    # Get the 2D slice and free parameter information
    slice_data, free_params = get_slice(results, fixed_params, output_key)
    
    # Check if we got a 2D slice
    if len(slice_data.shape) != 2:
        raise ValueError(f"Expected 2D slice, got shape {slice_data.shape}. You may need to fix more parameters.")
    
    # Get the names and values of the free parameters
    free_param_names = list(free_params.keys())
    if len(free_param_names) != 2:
        raise ValueError(f"Expected 2 free parameters, got {len(free_param_names)}")
    
    x_param = free_param_names[0]
    y_param = free_param_names[1]
    x_values = free_params[x_param]
    y_values = free_params[y_param]
    
    # Format fixed parameters for title
    fixed_params_str = ", ".join([f"{p}={v:.2f}" for p, v in fixed_params.items()])
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid(x_values, y_values)
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot heatmap
    im = ax.pcolormesh(X, Y, slice_data.T, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    
    # Add contours if requested
    if add_contours:
        if contour_levels is None:
            # Auto-generate levels
            min_val = np.nanmin(slice_data)
            max_val = np.nanmax(slice_data)
            contour_levels = np.linspace(min_val, max_val, 5)[1:-1]  # Skip min and max
        
        contour = ax.contour(X, Y, slice_data.T, levels=contour_levels, colors=contour_colors)
        ax.clabel(contour, inline=True, fontsize=8)
    
    # Set labels and title
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    
    if title is None:
        # Output key might be "final_state_S", extract just "S" for title
        if output_key.startswith("final_state_"):
            metric_name = output_key[len("final_state_"):]
            title = f"{results['model_name']} Model: {metric_name} Compartment"
        else:
            title = f"{results['model_name']} Model: {output_key}"
        
        if fixed_params:
            title += f" (Fixed: {fixed_params_str})"
    
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_1d_parameter_slice(
    results: Dict[str, Any],
    fixed_params: Dict[str, Union[float, int]],
    output_key: str = "r0",
    title: Optional[str] = None,
    color: str = "blue",
    fig_size: Tuple[int, int] = (10, 6),
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    marker: str = 'o',
    add_grid: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize a 1D slice from n-dimensional parameter sweep results
    
    Args:
        results: Results dictionary from sweep_n_parameters
        fixed_params: Dictionary mapping parameter names to fixed values
                     Must fix n-1 parameters to get a 1D slice
        output_key: The output to visualize
        title: Plot title (None for auto-generated)
        color: Line color
        fig_size: Figure size
        ymin, ymax: Y-axis limits (None for auto)
        marker: Point marker style
        add_grid: Whether to add grid lines
        save_path: Path to save figure (None to not save)
        
    Returns:
        Matplotlib figure
    """
    # Get the 1D slice and free parameter information
    slice_data, free_params = get_slice(results, fixed_params, output_key)
    
    # Check if we got a 1D slice
    if len(slice_data.shape) != 1:
        raise ValueError(f"Expected 1D slice, got shape {slice_data.shape}. You may need to fix more parameters.")
    
    # Get the name and values of the free parameter
    free_param_names = list(free_params.keys())
    if len(free_param_names) != 1:
        raise ValueError(f"Expected 1 free parameter, got {len(free_param_names)}")
    
    x_param = free_param_names[0]
    x_values = free_params[x_param]
    
    # Format fixed parameters for title
    fixed_params_str = ", ".join([f"{p}={v:.2f}" for p, v in fixed_params.items()])
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot line
    ax.plot(x_values, slice_data, marker=marker, color=color, linestyle='-')
    
    # Set labels and title
    ax.set_xlabel(x_param)
    
    # Set appropriate y-label based on the output key
    if output_key == "r0":
        ax.set_ylabel("Basic Reproduction Number (R0)")
    elif output_key == "homophily":
        ax.set_ylabel("Homophily Measure")
    elif output_key.startswith("final_state_"):
        compartment = output_key[len("final_state_"):]
        ax.set_ylabel(f"Final {compartment} Compartment")
    else:
        ax.set_ylabel(output_key)
    
    # Set y limits if specified
    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)
    
    if title is None:
        # Output key might be "final_state_S", extract just "S" for title
        if output_key.startswith("final_state_"):
            metric_name = output_key[len("final_state_"):]
            title = f"{results['model_name']} Model: {metric_name} Compartment"
        else:
            title = f"{results['model_name']} Model: {output_key}"
        
        if fixed_params:
            title += f" (Fixed: {fixed_params_str})"
    
    ax.set_title(title)
    
    # Add grid if requested
    if add_grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


