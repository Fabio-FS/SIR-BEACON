import jax
import jax.numpy as jnp
from typing import Dict, Any, Callable, Tuple, List, Optional, Union
import numpy as np

def batch_sweep(
    model_run_function: Callable,
    param_ranges: jnp.ndarray,
    base_params: Dict[str, Any],
    simulated_days: int,
    param_updater: Callable,  # Added param_updater function parameter
    initial_infected_prop: float = 1e-4,
    population_size: int = 100,
    use_contact_matrix: bool = False,
    batch_size: int = 1000,
    beta_params: Tuple[float, float] = (1.0, 1.0)
) -> Tuple:
    """
    Run a batch sweep over parameter ranges for any model
    
    Args:
        model_run_function: Function that runs a single simulation
        param_ranges: Array of parameter values to sweep over
        base_params: Base parameters for the model
        simulated_days: Number of days to simulate
        initial_infected_prop: Initial proportion of infected individuals
        population_size: Number of population compartments
        use_contact_matrix: Whether to use contact matrices
        batch_size: Batch size for processing parameter combinations
        beta_params: Default parameters for the beta distribution of population
        
    Returns:
        Tuple of (final_states, r0_values, homophily_values)
    """
    total_params = len(param_ranges)
    batch_size = min(batch_size, total_params)
    n_batches = (total_params + batch_size - 1) // batch_size
    results = []
    r0_results = []
    h_results = []
    
    # Import utility functions
    from ..utils.distributions import pol_mean_to_ab
    
    def run_single_sim(param_values):
        # Create a copy of base parameters
        sim_params = dict(base_params)
        
        # Initialize beta parameters
        sim_beta_params = beta_params
        
        # Extract parameter values based on their position in param_values
        # Use the provided parameter updater function
        updated_params, updated_beta_params = param_updater(sim_params, param_values, sim_beta_params)
        
        # Run the simulation
        return model_run_function(
            beta_params=updated_beta_params,
            params=updated_params,
            simulated_days=simulated_days,
            initial_infected_prop=initial_infected_prop,
            population_size=population_size,
            use_contact_matrix=use_contact_matrix
        )
    
    # Vectorize the simulation function to run batches efficiently
    vectorized_sim = jax.vmap(run_single_sim, in_axes=0)
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_params)
        batch_params = param_ranges[start_idx:end_idx]
        
        batch_states, batch_r0s, batch_hs = vectorized_sim(batch_params)
        results.append(batch_states)
        r0_results.append(batch_r0s)
        h_results.append(batch_hs)
    
    # Process results based on model output structure
    first_state = results[0]
    
    if isinstance(first_state, tuple):
        # Determine number of compartments based on first result
        n_compartments = len(first_state)
        
        # Concatenate results for each compartment
        final_states = tuple(
            jnp.concatenate([res[i] for res in results])
            for i in range(n_compartments)
        )
    else:
        # Single-compartment model
        final_states = jnp.concatenate(results)
    
    final_r0s = jnp.concatenate(r0_results)
    final_hs = jnp.concatenate(h_results)
    
    return final_states, final_r0s, final_hs


def create_parameter_grid(
    param1_range: Dict[str, float],
    param2_range: Dict[str, float]
) -> jnp.ndarray:
    """
    Create a parameter grid for sweeping over two parameters
    
    Args:
        param1_range: Dictionary with 'm' (min), 'M' (max), and 'n' (number of points)
        param2_range: Dictionary with 'm' (min), 'M' (max), and 'n' (number of points)
        
    Returns:
        Array of parameter pairs, shape (n_param1 * n_param2, 2)
    """
    from ..utils.distributions import homogeneous_distribution
    
    # Generate evenly spaced values for each parameter
    param1_vals = homogeneous_distribution(
        param1_range["n"], param1_range["m"], param1_range["M"]
    )
    param2_vals = homogeneous_distribution(
        param2_range["n"], param2_range["m"], param2_range["M"]
    )
    
    # Create a grid of all parameter combinations
    param1_mesh, param2_mesh = jnp.meshgrid(param1_vals, param2_vals)
    param_grid = jnp.stack([param1_mesh.ravel(), param2_mesh.ravel()], axis=1)
    
    return param_grid


def update_params_for_one_param(
    params: Dict[str, Any], 
    values: jnp.ndarray, 
    beta_params: Tuple[float, float],
    param_name: str
) -> Tuple[Dict[str, Any], Tuple[float, float]]:
    """
    Update parameters for a one-parameter sweep
    
    Args:
        params: Base parameters
        values: Parameter values array with a single value
        beta_params: Current beta distribution parameters
        param_name: Name of the parameter being swept
        
    Returns:
        Tuple of (updated_params, updated_beta_params)
    """
    from ..utils.distributions import pol_to_alpha, pol_mean_to_ab
    
    # Create a copy of params
    updated_params = dict(params)
    
    # Start with the provided beta_params
    updated_beta_params = beta_params
    
    # Extract the single parameter value
    param_value = values[0]
    
    # Group parameter names by type
    polarization_params = ["polarization", "beta_params"]
    
    # Handle polarization parameters
    if param_name in polarization_params:
        # Get polarization value
        pol = param_value
        
        # Get mean (use fixed_mean if provided, otherwise default to 0.5)
        mean = params.get("fixed_mean", 0.5)
        
        # Convert polarization and mean to alpha, beta parameters
        a, b = pol_mean_to_ab(pol, mean)
        updated_beta_params = (a, b)
    else:
        # For other parameters, just update them directly
        updated_params[param_name] = param_value
    
    return updated_params, updated_beta_params


def update_params_for_two_params(
    params: Dict[str, Any], 
    values: jnp.ndarray, 
    beta_params: Tuple[float, float],
    param1_name: str,
    param2_name: str
) -> Tuple[Dict[str, Any], Tuple[float, float]]:
    """
    Update parameters for a two-parameter sweep
    
    Args:
        params: Base parameters
        values: Parameter values array with two values
        beta_params: Current beta distribution parameters
        param1_name: Name of first parameter being swept
        param2_name: Name of second parameter being swept
        
    Returns:
        Tuple of (updated_params, updated_beta_params)
    """
    from ..utils.distributions import pol_to_alpha, pol_mean_to_ab
    
    # Create a copy of params
    updated_params = dict(params)
    
    # Start with the provided beta_params
    updated_beta_params = beta_params
    
    # Extract parameter values
    param1_value, param2_value = values[0], values[1]
    
    # Define parameter groups for cleaner checks
    polarization_params = ["polarization", "beta_params"]
    mean_params = ["mean"]
    
    # CASE 1: Both parameters affect population distribution (polarization and mean)
    if (param1_name in polarization_params and param2_name in mean_params) or \
       (param2_name in polarization_params and param1_name in mean_params):
        # Extract polarization and mean values
        if param1_name in polarization_params:
            pol, mean = param1_value, param2_value
        else:
            mean, pol = param1_value, param2_value
        
        # Convert polarization and mean to alpha, beta parameters
        a, b = pol_mean_to_ab(pol, mean)
        updated_beta_params = (a, b)
    
    # CASE 2: One parameter is polarization
    elif param1_name in polarization_params or param2_name in polarization_params:
        # Extract polarization and set the other parameter
        if param1_name in polarization_params:
            pol = param1_value
            updated_params[param2_name] = param2_value
        else:
            pol = param2_value
            updated_params[param1_name] = param1_value
        
        # Get mean (use fixed_mean if provided, otherwise default to 0.5)
        mean = params.get("fixed_mean", 0.5)
        
        # Convert polarization and mean to alpha, beta parameters
        a, b = pol_mean_to_ab(pol, mean)
        updated_beta_params = (a, b)
    
    # CASE 3: Other parameter combinations (no population distribution effects)
    else:
        # Set parameters directly
        updated_params[param1_name] = param1_value
        updated_params[param2_name] = param2_value
    
    return updated_params, updated_beta_params


def sweep_one_parameter(
    model_module: Any,
    param_name: str,
    param_range: Dict[str, float],
    custom_base_params: Optional[Dict[str, Any]] = None,
    simulated_days: int = 1000,
    initial_infected_prop: float = 1e-4,
    population_size: int = 100,
    use_contact_matrix: bool = False,
    batch_size: int = 1000,
    beta_params: Tuple[float, float] = (1.0, 1.0)
) -> Dict[str, Any]:
    """
    Run a sweep over a single parameter for any model
    
    Args:
        model_module: The imported model module (e.g., from models import SIRM)
        param_name: Name of the parameter to sweep
        param_range: Range for the parameter {"m": min, "M": max, "n": num_points}
        custom_base_params: Custom base parameters (if None, use model defaults)
        simulated_days: Number of simulated days
        initial_infected_prop: Initial proportion of infected individuals
        population_size: Number of population compartments
        use_contact_matrix: Whether to use contact matrices
        batch_size: Batch size for processing parameter combinations
        beta_params: Default parameters for the beta distribution of population
        
    Returns:
        Dictionary containing:
        - "final_state": Dictionary with compartment names as keys
        - "r0": R0 values for each parameter value
        - "homophily": Homophily values for each parameter value
        - Other metadata for visualization
    """
    # Get default parameters and update with custom ones if provided
    base_params = model_module.get_default_params()
    if custom_base_params is not None:
        base_params.update(custom_base_params)
    
    # Create parameter array (just a single column for one parameter)
    from ..utils.distributions import homogeneous_distribution
    param_vals = homogeneous_distribution(param_range["n"], param_range["m"], param_range["M"])
    param_grid = jnp.reshape(param_vals, (-1, 1))  # Convert to column vector
    
    # Define parameter updater function specifically for this sweep
    def update_params_for_sim(params, values, default_beta_params):
        return update_params_for_one_param(params, values, default_beta_params, param_name)
    
    # Run batch sweep
    final_states, r0s, hs = batch_sweep(
        model_run_function=model_module.run_simulation,
        param_ranges=param_grid,
        base_params=base_params,
        simulated_days=simulated_days,
        param_updater=update_params_for_sim,  # Pass the updater function
        initial_infected_prop=initial_infected_prop,
        population_size=population_size,
        use_contact_matrix=use_contact_matrix,
        batch_size=batch_size,
        beta_params=beta_params
    )
    
    # Get model name and compartment names
    model_name, compartment_names = model_module.get_compartment_info()
    
    # Reshape param_vals for easier plotting
    n = param_range['n']
    
    # Organize the final states into a dictionary with compartment names as keys
    final_state_dict = {}
    for i, name in enumerate(compartment_names):
        if isinstance(final_states, tuple) and i < len(final_states):
            # For each compartment, organize the data
            compartment_data = final_states[i]
            
            # If compartment_data is a matrix, keep its structure
            if len(compartment_data.shape) == 2:
                final_state_dict[name] = compartment_data
            else:
                # If it's already a 3D array, just store it
                final_state_dict[name] = compartment_data
    
    # Return results in a structured dictionary
    return {
        "model_name": model_name,
        "final_state": final_state_dict,
        "r0": r0s,
        "homophily": hs,
        "parameter_values": param_vals,
        "parameter_name": param_name
    }


def sweep_two_parameters(
    model_module: Any,
    param1_name: str,
    param1_range: Union[Dict[str, float], List[float], np.ndarray],
    param2_name: str,
    param2_range: Union[Dict[str, float], List[float], np.ndarray],
    custom_base_params: Optional[Dict[str, Any]] = None,
    simulated_days: int = 1000,
    initial_infected_prop: float = 1e-4,
    population_size: int = 100,
    use_contact_matrix: bool = False,
    batch_size: int = 1000,
    beta_params: Tuple[float, float] = (1.0, 1.0)
) -> Dict[str, Any]:
    """
    Run a sweep over two parameters for any model
    
    Args:
        model_module: The imported model module (e.g., from models import SIRM)
        param1_name: Name of the first parameter to sweep
        param1_range: Range for the first parameter - either a dictionary {"m": min, "M": max, "n": num_points}
                     or a list/array of specific values to use
        param2_name: Name of the second parameter to sweep
        param2_range: Range for the second parameter - either a dictionary {"m": min, "M": max, "n": num_points}
                     or a list/array of specific values to use
        custom_base_params: Custom base parameters (if None, use model defaults)
        simulated_days: Number of simulated days
        initial_infected_prop: Initial proportion of infected individuals
        population_size: Number of population compartments
        use_contact_matrix: Whether to use contact matrices
        batch_size: Batch size for processing parameter combinations
        beta_params: Default parameters for the beta distribution of population
        
    Returns:
        Dictionary containing:
        - "final_state": Dictionary with compartment names as keys
        - "r0": R0 values for each parameter combination
        - "homophily": Homophily values for each parameter combination
        - Other metadata for visualization
    """
    # Get default parameters and update with custom ones if provided
    base_params = model_module.get_default_params()
    if custom_base_params is not None:
        base_params.update(custom_base_params)
    
    # Process parameter ranges based on input type
    def process_param_range(param_range):
        from ..utils.distributions import homogeneous_distribution
        
        if isinstance(param_range, dict):
            # Dictionary format: use homogeneous_distribution
            values = homogeneous_distribution(
                param_range["n"], param_range["m"], param_range["M"]
            )
            param_dict = param_range.copy()  # Keep the original dictionary for metadata
            return values, param_dict
        else:
            # List/array format: convert to array if needed
            values = jnp.array(param_range)
            # Create equivalent dictionary for metadata
            param_dict = {
                "m": float(values.min()),
                "M": float(values.max()),
                "n": len(values)
            }
            return values, param_dict
    
    # Process parameter ranges
    param1_vals, param1_dict = process_param_range(param1_range)
    param2_vals, param2_dict = process_param_range(param2_range)
    
    # Create parameter grid
    param1_mesh, param2_mesh = jnp.meshgrid(param1_vals, param2_vals)
    param_grid = jnp.stack([param1_mesh.ravel(), param2_mesh.ravel()], axis=1)
    
    # Define the update function specific to this two-parameter sweep
    def update_params_for_sim(params, values, default_beta_params):
        return update_params_for_two_params(
            params, values, default_beta_params, param1_name, param2_name
        )
    
    # Run batch sweep
    final_states, r0s, hs = batch_sweep(
        model_run_function=model_module.run_simulation,
        param_ranges=param_grid,
        base_params=base_params,
        simulated_days=simulated_days,
        param_updater=update_params_for_sim,  # Pass the updater function
        initial_infected_prop=initial_infected_prop,
        population_size=population_size,
        use_contact_matrix=use_contact_matrix,
        batch_size=batch_size,
        beta_params=beta_params
    )
    
    # Get model name and compartment names
    model_name, compartment_names = model_module.get_compartment_info()
    
    # Get dimensions for reshaping
    n1 = param1_dict['n']
    n2 = param2_dict['n']
    
    # Reshape parameter grid values for easier plotting
    param1_vals_grid = param_grid[:, 0].reshape(n2, n1)
    param2_vals_grid = param_grid[:, 1].reshape(n2, n1)
    
    # Reshape R0 and homophily values
    r0s_grid = r0s.reshape(n2, n1)
    hs_grid = hs.reshape(n2, n1)
    
    # Organize the final states into a dictionary with compartment names as keys
    final_state_dict = {}
    for i, name in enumerate(compartment_names):
        if isinstance(final_states, tuple) and i < len(final_states):
            # For each compartment, reshape the data to match parameter grid
            compartment_data = final_states[i]
            
            # If compartment_data is a matrix, reshape it properly
            if len(compartment_data.shape) == 2:
                # Reshape to (n2, n1, population_size)
                reshaped_data = compartment_data.reshape(n2, n1, -1)
                final_state_dict[name] = reshaped_data
            else:
                # If it's already a 3D array, just store it
                final_state_dict[name] = compartment_data
    
    # Return results in a structured dictionary
    return {
        "model_name": model_name,
        "final_state": final_state_dict,
        "r0": r0s_grid,
        "homophily": hs_grid,
        "parameter_grid": {
            "param1_vals": param1_vals_grid,
            "param2_vals": param2_vals_grid
        },
        "parameter_names": {
            "param1": param1_name,
            "param2": param2_name
        },
        "parameter_ranges": {
            "param1": param1_dict,
            "param2": param2_dict
        }
    }


def sweep_n_parameters(
    model_module: Any,
    param_specs: Dict[str, Dict[str, float]],
    custom_base_params: Optional[Dict[str, Any]] = None,
    simulated_days: int = 1000,
    initial_infected_prop: float = 1e-4,
    population_size: int = 100,
    use_contact_matrix: bool = False,
    batch_size: int = 1000,
    beta_params: Tuple[float, float] = (1.0, 1.0)
) -> Dict[str, Any]:
    """
    Run a sweep over n parameters for any model
    
    Args:
        model_module: The imported model module (e.g., from models import SIRM)
        param_specs: Dictionary of parameter specifications, where:
                     - Keys are parameter names
                     - Values are dictionaries with "m" (min), "M" (max), "n" (num_points)
        custom_base_params: Custom base parameters (if None, use model defaults)
        simulated_days: Number of simulated days
        initial_infected_prop: Initial proportion of infected individuals
        population_size: Number of population compartments
        use_contact_matrix: Whether to use contact matrices
        batch_size: Batch size for processing parameter combinations
        beta_params: Default parameters for the beta distribution of population
        
    Returns:
        Dictionary containing:
        - "final_state": Dictionary with compartment names as keys
        - "r0": R0 values for each parameter combination
        - "homophily": Homophily values for each parameter combination
        - Parameter grid metadata for analysis
    """
    import itertools
    from ..utils.distributions import homogeneous_distribution, pol_mean_to_ab
    
    # Get default parameters and update with custom ones if provided
    base_params = model_module.get_default_params()
    if custom_base_params is not None:
        base_params.update(custom_base_params)
    
    # Extract parameter names and create grid points for each parameter
    param_names = list(param_specs.keys())
    param_values = []
    param_sizes = []
    
    for param_name, param_range in param_specs.items():
        # Generate evenly spaced values for each parameter
        values = homogeneous_distribution(
            param_range["n"], param_range["m"], param_range["M"]
        )
        param_values.append(values)
        param_sizes.append(param_range["n"])
    
    # Generate all combinations of parameter values
    param_combinations = list(itertools.product(*param_values))
    param_grid = jnp.array(param_combinations)
    
    # Define parameter groups for cleaner checks
    polarization_params = ["polarization", "beta_params"]
    
    # Define the update function for this n-parameter sweep
    def update_params_for_sim(params, values, default_beta_params):
        # Create a copy of params
        updated_params = dict(params)
        
        # Start with the provided beta_params
        updated_beta_params = default_beta_params
        
        # Track if we've already handled polarization
        polarization_handled = False
        
        # Handle each parameter
        for i, param_name in enumerate(param_names):
            if param_name in polarization_params and not polarization_handled:
                # Get polarization value
                pol = values[i]
                
                # Get mean (use fixed_mean if provided, otherwise default to 0.5)
                mean = params.get("fixed_mean", 0.5)
                
                # Convert polarization and mean to alpha, beta parameters
                a, b = pol_mean_to_ab(pol, mean)
                updated_beta_params = (a, b)
                
                # Mark polarization as handled
                polarization_handled = True
            else:
                # For other parameters, just update them directly
                updated_params[param_name] = values[i]
        
        return updated_params, updated_beta_params
    
    # Run batch sweep
    final_states, r0s, hs = batch_sweep(
        model_run_function=model_module.run_simulation,
        param_ranges=param_grid,
        base_params=base_params,
        simulated_days=simulated_days,
        param_updater=update_params_for_sim,  # Pass the updater function
        initial_infected_prop=initial_infected_prop,
        population_size=population_size,
        use_contact_matrix=use_contact_matrix,
        batch_size=batch_size,
        beta_params=beta_params
    )
    
    # Get model name and compartment names
    model_name, compartment_names = model_module.get_compartment_info()
    
    # Organize the final states into a dictionary with compartment names as keys
    final_state_dict = {}
    for i, name in enumerate(compartment_names):
        if isinstance(final_states, tuple) and i < len(final_states):
            # Store compartment data directly - will be a 2D array where:
            # - Each row corresponds to a parameter combination
            # - Each column corresponds to a population compartment
            final_state_dict[name] = final_states[i]
    
    # Shape information for reshaping results to n-dimensional arrays if needed
    shape_info = {
        "param_names": param_names,
        "param_sizes": param_sizes,
        "original_shape": param_sizes,
        "flattened_shape": r0s.shape
    }
    
    # Build parameter information for each parameter
    param_info = {}
    for i, param_name in enumerate(param_names):
        param_info[param_name] = {
            "values": param_values[i],
            "min": param_specs[param_name]["m"],
            "max": param_specs[param_name]["M"],
            "n": param_specs[param_name]["n"]
        }
    
    # Return results in a structured dictionary
    return {
        "model_name": model_name,
        "final_state": final_state_dict,
        "r0": r0s,
        "homophily": hs,
        "parameter_grid": param_grid,
        "parameter_info": param_info,
        "shape_info": shape_info,
        "n_parameters": len(param_names)
    }


def run_parameter_sweep(
    model_module: Any,
    param_name: str,
    param_range: Dict[str, float],
    fixed_params: Dict[str, Any],
    simulated_days: int = 1000,
    population_size: int = 100,
    beta_params: Tuple[float, float] = (1.0, 1.0)
) -> Dict[str, Any]:
    """
    Run a parameter sweep for a single parameter while keeping others fixed.
    
    Args:
        model_module: The imported model module (e.g., from models import SIRM)
        param_name: Name of the parameter to sweep
        param_range: Range for the parameter {"m": min, "M": max, "n": num_points}
        fixed_params: Dictionary of fixed parameters
        simulated_days: Number of simulated days
        population_size: Number of population compartments
        beta_params: Tuple of beta distribution parameters (for polarization)
        
    Returns:
        Dictionary containing sweep results
    """
    # Run the sweep
    result = sweep_one_parameter(
        model_module=model_module,
        param_name=param_name,
        param_range=param_range,
        custom_base_params=fixed_params,
        simulated_days=simulated_days,
        population_size=population_size,
        beta_params=beta_params
    )
    
    return result