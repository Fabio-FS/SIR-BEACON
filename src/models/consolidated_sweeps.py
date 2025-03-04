
import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple, Callable
from ..utils.distributions import pol_mean_to_ab, pol_to_alpha, homogeneous_distribution
from ..utils.consolidated_batch_sweep import consolidated_batch_sweep
from .consolidated_dynamics import sim_maskSIR_final, sim_SIRT_final, sim_SIRV_final

# Parameter updater functions
def update_mask_pol_params(params: Dict[str, Any], values: jnp.ndarray) -> Dict[str, Any]:
    """Updates parameters with mask-wearing and alpha values"""
    params['mu_max'] = values[0]
    params['beta_params'] = (values[1], values[1])
    return params

def update_test_pol_params(params: Dict[str, Any], values: jnp.ndarray) -> Dict[str, Any]:
    """Updates parameters with testing rate and alpha values"""
    params['testing_rates'] = (0, values[0])
    params['beta_params'] = (values[1], values[1])
    return params

def update_vacc_pol_params(params: Dict[str, Any], values: jnp.ndarray) -> Dict[str, Any]:
    """Updates parameters with vaccination rate and alpha values"""
    params['vaccination_rates'] = (0, values[0])
    params['beta_params'] = (values[1], values[1])
    return params

def update_pol_mean_params(params: Dict[str, Any], values: jnp.ndarray) -> Dict[str, Any]:
    """Updates parameters with polarization and mean values"""
    alpha, beta = pol_mean_to_ab(values[0], values[1])
    params['beta_params'] = (alpha, beta)
    return params

def update_h_pol_params(params: Dict[str, Any], values: jnp.ndarray) -> Dict[str, Any]:
    """Updates parameters with homophily and alpha values"""
    params['homophilic_tendency'] = values[0]
    params['beta_params'] = (values[1], values[1])
    return params

# Sweep functions
def sweep_pol_behavior(
    model_type: str,
    # Accept all the possible parameter names used by different models
    mask_max_range: dict = None,
    test_max_range: dict = None,
    vacc_max_range: dict = None,
    behavior_max_range: dict = None,
    pol_range: dict = None,
    h: float = 0,
    dT: float = 0.25,
    T: int = 1000,
    recovery_rate: float = 0.1,
    susceptibility_rate: float = 0.6,
    beta_M: float = 0.6,
    batch_size: int = 1000,
    use_contact_matrix: bool = False,
    initial_infected_prop: float = 1e-4,
    SPB_exponent: float = 1.0,
    N_COMPARTMENTS: int = 100
) -> Tuple:
    """Sweep over polarization and behavior intensity for different model types"""
    # Convert model-specific range parameters to a unified parameter
    if behavior_max_range is None:
        if model_type == "mask" and mask_max_range is not None:
            behavior_max_range = mask_max_range
        elif model_type == "test" and test_max_range is not None:
            behavior_max_range = test_max_range
        elif model_type == "vaccine" and vacc_max_range is not None:
            behavior_max_range = vacc_max_range
        else:
            # Default if no range is provided
            behavior_max_range = {"m": 0, "M": 1, "n": 10}
    
    # Set up base parameters based on model type
    base_params = {
        'recovery_rate': recovery_rate,
        'dT': dT,
        'homophilic_tendency': h,
        'beta_params': (2.0, 2.0),
        'SPB_exponent': SPB_exponent
    }
    
    if model_type == "mask":
        base_params.update({
            'beta_M': beta_M,
            'mu_max': 0.8,  # Default, will be overwritten by sweep
        })
        simulation_fn = sim_maskSIR_final
        param_updater = update_mask_pol_params
    elif model_type == "test":
        base_params.update({
            'susceptibility_rate': susceptibility_rate,
            'testing_rates': (0.0, 0.05),  # Default, will be overwritten by sweep
        })
        simulation_fn = sim_SIRT_final
        param_updater = update_test_pol_params
    elif model_type == "vaccine":
        base_params.update({
            'susceptibility_rate': susceptibility_rate,
            'vaccination_rates': (0.0, 0.05),  # Default, will be overwritten by sweep
        })
        simulation_fn = sim_SIRV_final
        param_updater = update_vacc_pol_params
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Prepare the parameter ranges
    behavior_vals = homogeneous_distribution(behavior_max_range["n"], behavior_max_range["m"], behavior_max_range["M"])
    pol_vals = homogeneous_distribution(pol_range["n"], pol_range["m"], pol_range["M"])
    alpha_vals = pol_to_alpha(pol_vals)
    
    behavior_mesh, alpha_mesh = jnp.meshgrid(behavior_vals, alpha_vals)
    param_ranges = jnp.stack([behavior_mesh.ravel(), alpha_mesh.ravel()], axis=1)
    
    n_steps = int(T / dT)
    
    # Run the batch sweep
    return consolidated_batch_sweep(
        simulation_fn=simulation_fn,
        param_ranges=param_ranges,
        base_params=base_params,
        n_steps=n_steps,
        batch_size=batch_size,
        param_updater=param_updater,
        initial_infected_prop=initial_infected_prop,
        N_COMPARTMENTS=N_COMPARTMENTS,
        use_contact_matrix=use_contact_matrix,
        SPB_exponent=SPB_exponent
    )

def sweep_pol_mean(
    model_type: str,
    pol_range: dict,
    mean_range: dict,
    h: float = 0,
    dT: float = 1,
    T: int = 1000,
    recovery_rate: float = 0.1,
    susceptibility_rate: float = 0.6,
    beta_M: float = 0.6,
    behavior_rate: float = 0.05,  # Unified parameter for test/vacc/mask rate
    # Add model-specific parameters
    mu_max: float = None,         # For mask model
    test_rate: float = None,      # For test model
    vaccination_rate: float = None, # For vaccine model
    batch_size: int = 1000,
    use_contact_matrix: bool = False,
    initial_infected_prop: float = 1e-4,
    SPB_exponent: float = 1.0,
    N_COMPARTMENTS: int = 100
) -> Tuple:
    """Sweep over polarization and mean values for different model types"""
    # Set up base parameters based on model type
    base_params = {
        'recovery_rate': recovery_rate,
        'dT': dT,
        'homophilic_tendency': h,
        'beta_params': (1.0, 1.0),
        'SPB_exponent': SPB_exponent
    }
    
    # Handle model-specific rate parameters
    if model_type == "mask":
        actual_behavior_rate = mu_max if mu_max is not None else behavior_rate
        base_params.update({
            'beta_M': beta_M,
            'mu_max': actual_behavior_rate,
        })
        simulation_fn = sim_maskSIR_final
    elif model_type == "test":
        actual_behavior_rate = test_rate if test_rate is not None else behavior_rate
        base_params.update({
            'susceptibility_rate': susceptibility_rate,
            'testing_rates': (0.0, actual_behavior_rate),
        })
        simulation_fn = sim_SIRT_final
    elif model_type == "vaccine":
        actual_behavior_rate = vaccination_rate if vaccination_rate is not None else behavior_rate
        base_params.update({
            'susceptibility_rate': susceptibility_rate,
            'vaccination_rates': (0.0, actual_behavior_rate),
        })
        simulation_fn = sim_SIRV_final
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Prepare the parameter ranges
    pol_vals = homogeneous_distribution(pol_range["n"], pol_range["m"], pol_range["M"])
    mean_vals = homogeneous_distribution(mean_range["n"], mean_range["m"], mean_range["M"])
    
    pol_mesh, mean_mesh = jnp.meshgrid(pol_vals, mean_vals)
    param_ranges = jnp.stack([pol_mesh.ravel(), mean_mesh.ravel()], axis=1)
    
    n_steps = int(T / dT)
    
    # Run the batch sweep
    return consolidated_batch_sweep(
        simulation_fn=simulation_fn,
        param_ranges=param_ranges,
        base_params=base_params,
        n_steps=n_steps,
        batch_size=batch_size,
        param_updater=update_pol_mean_params,
        initial_infected_prop=initial_infected_prop,
        N_COMPARTMENTS=N_COMPARTMENTS,
        use_contact_matrix=use_contact_matrix,
        SPB_exponent=SPB_exponent
    )

def sweep_hom_pol(
    model_type: str,
    h_range: dict,
    pol_range: dict,
    dT: float = 1,
    T: int = 1000,
    recovery_rate: float = 0.1,
    susceptibility_rate: float = 0.6,
    beta_M: float = 0.6,
    behavior_rate: float = 0.05,  # Unified parameter for test/vacc/mask rate
    # Add model-specific parameters
    mu_max: float = None,         # For mask model
    test_rate: float = None,      # For test model
    vaccination_rate: float = None, # For vaccine model
    batch_size: int = 1000,
    initial_infected_prop: float = 1e-4,
    SPB_exponent: float = 1.0,
    use_contact_matrix: bool = True,
    N_COMPARTMENTS: int = 100
) -> Tuple:
    """Sweep over homophily and polarization values for different model types"""
    # Set up base parameters based on model type
    base_params = {
        'recovery_rate': recovery_rate,
        'dT': dT,
        'beta_params': (2.0, 2.0),
        'SPB_exponent': SPB_exponent
    }
    
    # Handle model-specific rate parameters
    if model_type == "mask":
        actual_behavior_rate = mu_max if mu_max is not None else behavior_rate
        base_params.update({
            'beta_M': beta_M,
            'mu_max': actual_behavior_rate,
        })
        simulation_fn = sim_maskSIR_final
    elif model_type == "test":
        actual_behavior_rate = test_rate if test_rate is not None else behavior_rate
        base_params.update({
            'susceptibility_rate': susceptibility_rate,
            'testing_rates': (0.0, actual_behavior_rate),
        })
        simulation_fn = sim_SIRT_final
    elif model_type == "vaccine":
        actual_behavior_rate = vaccination_rate if vaccination_rate is not None else behavior_rate
        base_params.update({
            'susceptibility_rate': susceptibility_rate,
            'vaccination_rates': (0.0, actual_behavior_rate),
        })
        simulation_fn = sim_SIRV_final
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Prepare the parameter ranges
    h_vals = homogeneous_distribution(h_range["n"], h_range["m"], h_range["M"])
    pol_vals = homogeneous_distribution(pol_range["n"], pol_range["m"], pol_range["M"])
    alpha_vals = pol_to_alpha(pol_vals)
    
    h_mesh, alpha_mesh = jnp.meshgrid(h_vals, alpha_vals)
    param_ranges = jnp.stack([h_mesh.ravel(), alpha_mesh.ravel()], axis=1)
    
    n_steps = int(T / dT)
    
    # Run the batch sweep
    return consolidated_batch_sweep(
        simulation_fn=simulation_fn,
        param_ranges=param_ranges,
        base_params=base_params,
        n_steps=n_steps,
        batch_size=batch_size,
        param_updater=update_h_pol_params,
        initial_infected_prop=initial_infected_prop,
        N_COMPARTMENTS=N_COMPARTMENTS,
        use_contact_matrix=use_contact_matrix,
        SPB_exponent=SPB_exponent
    )