# File: src/utils/sensitivity/core.py
"""
Functions for analyzing parameter sensitivity in epidemic models
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional, Tuple

def get_slice(results: Dict[str, Any], 
             fixed_params: Dict[str, float], 
             output_key: str = "r0") -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Extract a 1D or 2D slice from n-dimensional results by fixing n-2 or n-1 parameters
    
    Args:
        results: Results dictionary from sweep_n_parameters
        fixed_params: Dictionary mapping parameter names to fixed values (index or actual value)
                     You must fix n-2 parameters to get a 2D slice, or n-1 for a 1D slice
        output_key: The output to extract (e.g., "r0", "homophily", "final_state_S", "infections")
        
    Returns:
        Tuple of (slice_data, free_param_info) where:
        - slice_data is the extracted slice 
        - free_param_info contains the remaining parameter names and their values
    """
    # Handle special case for "infections" metric which needs to be calculated
    if output_key == "infections":
        # First, reshape the required compartments
        reshape_keys = []
        
        # Get all compartment data needed for infection calculation
        if "final_state" in results:
            for compartment in ["S", "V"]:
                if compartment in results["final_state"]:
                    reshape_keys.append(f"final_state_{compartment}")
        
        if not reshape_keys:
            raise ValueError("Cannot calculate infections: no S or V compartments found in results")
        
        # Reshape the components needed for infection calculation
        reshaped_components = reshape_results(results, reshape_keys)
        
        # Create a new reshaped results dictionary
        reshaped_results = {}
        for key, value in reshaped_components.items():
            reshaped_results[key] = value
        
        # Calculate infections as 1 - S - V
        total_shape = reshaped_results["final_state_S"].shape
        infections = np.ones(total_shape)
        
        # Subtract final susceptible population
        if "final_state_S" in reshaped_results:
            infections -= reshaped_results["final_state_S"]
            
        # Subtract final vaccinated population if it exists
        if "final_state_V" in reshaped_results:
            infections -= reshaped_results["final_state_V"]
            
        # Add the infections data to reshaped results
        reshaped_results["infections"] = infections
    else:
        # Original behavior for other metrics
        reshape_keys = [output_key]
        if output_key.startswith("final_state_") and output_key not in results:
            # Handle compartment data
            compartment = output_key[len("final_state_"):]
            if "final_state" in results and compartment in results["final_state"]:
                # This is a valid compartment, proceed with reshaping
                pass
            else:
                raise ValueError(f"Compartment '{compartment}' not found in results")
                
        reshaped_results = reshape_results(results, reshape_keys)
    
    if output_key not in reshaped_results:
        raise ValueError(f"Output key '{output_key}' not found in results after reshaping")
    
    # Get parameter information
    n_params = results["n_parameters"]
    param_names = results["shape_info"]["param_names"]
    param_info = results["parameter_info"]
    
    # Check if we're fixing the right number of parameters
    n_fixed = len(fixed_params)
    if n_fixed not in [n_params-1, n_params-2]:
        raise ValueError(f"For {n_params}D data, you must fix {n_params-2} parameters for a 2D slice or {n_params-1} for a 1D slice")
    
    # Convert parameter values to indices if actual values are provided
    fixed_indices = {}
    for param_name, value in fixed_params.items():
        if param_name not in param_names:
            raise ValueError(f"Parameter '{param_name}' not found in sweep parameters")
        
        idx = param_names.index(param_name)
        
        # If value is an index, use it directly
        if isinstance(value, int) and 0 <= value < param_info[param_name]["n"]:
            fixed_indices[idx] = value
        else:
            # Find the closest value in the parameter values array
            param_values = param_info[param_name]["values"]
            closest_idx = jnp.argmin(jnp.abs(param_values - value))
            fixed_indices[idx] = int(closest_idx)
    
    # Identify free parameters (those not fixed)
    free_params = {}
    for i, param_name in enumerate(param_names):
        if param_name not in fixed_params:
            free_params[param_name] = param_info[param_name]["values"]
    
    # Create a slice indexer
    slicer = []
    for i in range(n_params):
        if i in fixed_indices:
            slicer.append(fixed_indices[i])
        else:
            slicer.append(slice(None))
    
    # Extract the slice
    result_slice = reshaped_results[output_key][tuple(slicer)]
    
    return result_slice, free_params

def reshape_results(results: Dict[str, Any], output_keys: list = None) -> Dict[str, Any]:
    """
    Reshape flat results from n-parameter sweep into n-dimensional arrays for analysis
    
    Args:
        results: Results dictionary from sweep_n_parameters
        output_keys: List of result keys to reshape (None means reshape all applicable keys)
        
    Returns:
        Dictionary with reshaped arrays for each key
    """
    # Default keys to reshape if none specified
    if output_keys is None:
        output_keys = ["r0", "homophily"]
        
        # Add all final state compartments if they exist
        if "final_state" in results:
            for compartment in results["final_state"].keys():
                output_keys.append(f"final_state_{compartment}")
    
    shape_info = results["shape_info"]
    param_sizes = shape_info["param_sizes"]
    reshaped_results = {}
    
    # Reshape each requested output
    for key in output_keys:
        if key.startswith("final_state_") and "final_state" in results:
            # Handle final state compartments
            compartment = key[len("final_state_"):]
            if compartment in results["final_state"]:
                # Sum across population compartments to get total for each parameter combination
                data = jnp.sum(results["final_state"][compartment], axis=1)
                reshaped_results[key] = jnp.reshape(data, param_sizes)
        elif key in results:
            # Handle standard outputs like r0, homophily
            reshaped_results[key] = jnp.reshape(results[key], param_sizes)
    
    return reshaped_results

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