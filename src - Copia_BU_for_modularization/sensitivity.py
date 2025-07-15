# File: src/utils/visualization/sensitivity.py
"""
Functions for analyzing parameter sensitivity
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional, Tuple

from .slices import get_slice

def analyze_parameter_sensitivity(
    results: Dict[str, Any],
    output_key: str = "r0",
    n_points: int = 5,
    baseline_values: Optional[Dict[str, float]] = None,
    plot: bool = True,
    fig_size: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze and visualize parameter sensitivity from n-parameter sweep
    
    Args:
        results: Results dictionary from sweep_n_parameters
        output_key: The output to analyze (e.g., "r0", "final_state_S")
        n_points: Number of sample points along each parameter dimension
        baseline_values: Dictionary with baseline values for each parameter
                        (None to use midpoint of each parameter range)
        plot: Whether to create a plot
        fig_size: Figure size for the plot
        save_path: Path to save the plot
        
    Returns:
        Dictionary with sensitivity measures for each parameter
    """
    param_names = results["shape_info"]["param_names"]
    param_info = results["parameter_info"]
    n_params = len(param_names)
    
    # Prepare baseline values if not provided
    if baseline_values is None:
        baseline_values = {}
        for param_name in param_names:
            min_val = param_info[param_name]["min"]
            max_val = param_info[param_name]["max"]
            baseline_values[param_name] = (min_val + max_val) / 2
    
    # Check if we have all baseline values
    for param_name in param_names:
        if param_name not in baseline_values:
            min_val = param_info[param_name]["min"]
            max_val = param_info[param_name]["max"]
            baseline_values[param_name] = (min_val + max_val) / 2
    
    # Calculate sensitivity for each parameter
    sensitivity_data = {}
    
    for param_name in param_names:
        # Create fixed parameters from baseline but exclude current parameter
        fixed_params = dict(baseline_values)
        del fixed_params[param_name]
        
        # Get 1D slice for the current parameter with others fixed at baseline
        slice_data, free_params = get_slice(results, fixed_params, output_key)
        
        # Get parameter values
        param_values = free_params[param_name]
        
        # Calculate min, max, range, and normalized sensitivity
        min_val = np.nanmin(slice_data)
        max_val = np.nanmax(slice_data)
        range_val = max_val - min_val
        
        # Calculate normalized sensitivity (range / parameter_range)
        param_range = param_info[param_name]["max"] - param_info[param_name]["min"]
        normalized_sensitivity = range_val / param_range if param_range != 0 else 0
        
        # Store sensitivity metrics
        sensitivity_data[param_name] = {
            "values": param_values,
            "output": slice_data,
            "min": min_val,
            "max": max_val,
            "range": range_val,
            "normalized_sensitivity": normalized_sensitivity
        }
    
    # Create visualization if requested
    if plot:
        # Sort parameters by sensitivity
        sorted_params = sorted(
            param_names,
            key=lambda p: sensitivity_data[p]["normalized_sensitivity"],
            reverse=True
        )
        
        # Create figure with one subplot per parameter
        fig, axes = plt.subplots(n_params, 1, figsize=fig_size, sharex=False, sharey=True)
        if n_params == 1:
            axes = [axes]  # Make iterable for single parameter case
        
        for i, param_name in enumerate(sorted_params):
            ax = axes[i]
            data = sensitivity_data[param_name]
            
            # Plot parameter effect
            ax.plot(
                data["values"],
                data["output"],
                'o-',
                color=f"C{i}",
                label=f"Sensitivity: {data['normalized_sensitivity']:.4f}"
            )
            
            # Mark baseline value
            baseline_idx = np.argmin(np.abs(data["values"] - baseline_values[param_name]))
            baseline_value = data["values"][baseline_idx]
            baseline_output = data["output"][baseline_idx]
            ax.plot(
                [baseline_value],
                [baseline_output],
                'ro',
                markersize=8,
                label=f"Baseline: {baseline_value:.2f}"
            )
            
            # Add grid and legend
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='best')
            
            # Add parameter name
            ax.set_ylabel(param_name)
            
            # Add y-label only to middle plot
            if i == n_params // 2:
                ax.set_ylabel(f"{param_name}\n\n{output_key}", fontsize=12)
            else:
                ax.set_ylabel(param_name)
            
            # Remove x-label from all but the last plot
            if i < n_params - 1:
                ax.set_xticklabels([])
        
        # Add overall title
        if output_key.startswith("final_state_"):
            metric_name = output_key[len("final_state_"):]
            title = f"{results['model_name']} Model: Parameter Sensitivity for {metric_name} Compartment"
        else:
            title = f"{results['model_name']} Model: Parameter Sensitivity for {output_key}"
            
        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for title
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return sensitivity_data