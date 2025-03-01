import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple
from .dynamic import sim_maskSIR_final
from .dynamic import StateType
from ...utils.batch_sweep import batch_sweep
from ...utils.distributions import pol_mean_to_ab, pol_to_alpha, homogeneous_distribution

def update_pol_mean_params(params: Dict[str, Any], values: jnp.ndarray) -> Dict[str, Any]:
    """Updates parameters with new alpha and beta values derived from variance and mean
    
    Args:
        params: Base parameter dictionary
        values: Array where:
            values[0]: variance value
            values[1]: mean value
    """
    alpha, beta = pol_mean_to_ab(values[0], values[1])
    params['beta_params'] = (alpha, beta)
    return params

def update_mask_pol_params(params: Dict[str, Any], values: jnp.ndarray) -> Dict[str, Any]:
    """Updates parameters with new maximum mask-wearing and alpha values
    
    Args:
        values[0]: maximum mask-wearing value (mu_max)
        values[1]: alpha value (already converted from polarization)
    """
    params['mu_max'] = values[0]           # maximum mask-wearing value
    params['beta_params'] = (values[1], values[1])      # both parameters equal
    return params

def update_h_pol_params(params: Dict[str, Any], values: jnp.ndarray) -> Dict[str, Any]:
    """Updates parameters with new homophily and alpha values
    
    Args:
        params: Base parameter dictionary
        values: Array where:
            values[0]: homophily value (h)
            values[1]: alpha value (already converted from polarization)
    """
    params['homophilic_tendency'] = values[0]
    params['beta_params'] = (values[1], values[1])   # both parameters equal for polarization
    return params

def sweep_pol_mean_maskSIR(
    pol_range: dict = {"m": 0, "M": 1, "n": 100},
    mean_range: dict = {"m": 0, "M": 1, "n": 100},
    h: float = 0,
    dT: float = 1,
    T: int = 1000,
    recovery_rate: float = 0.1,
    beta_M: float = 0.6,      # Maximum susceptibility
    mu_max: float = 1,      # Maximum mask-wearing value
    batch_size: int = 1000,
    use_contact_matrix: bool = False,
    initial_infected_prop: float = 1e-4,
    SPB_exponent: float = 1.0,
    N_COMPARTMENTS: int = 100
) -> Tuple[StateType, jnp.ndarray]:
    base_params = {
        'recovery_rate': recovery_rate,
        'dT': dT,
        'homophilic_tendency': h,
        'beta_M': beta_M,
        'mu_max': mu_max,
        'beta_params': (1.0, 1.0),
        'SPB_exponent': SPB_exponent
    }
    
    pol_vals = homogeneous_distribution(pol_range["n"], pol_range["m"], pol_range["M"])
    mean_vals = homogeneous_distribution(mean_range["n"], mean_range["m"], mean_range["M"])
    
    pol_mesh, mean_mesh = jnp.meshgrid(pol_vals, mean_vals)
    param_ranges = jnp.stack([pol_mesh.ravel(), mean_mesh.ravel()], axis=1)
    
    n_steps = int(T / dT)
    
    return batch_sweep(
        simulation_fn=sim_maskSIR_final,
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

def sweep_pol_mask_maskSIR(
    mask_max_range: dict = {"m": 0, "M": 1, "n": 10},
    pol_range: dict = {"m": 0, "M": 1, "n": 10},
    h: float = 0,
    dT: float = 0.25,
    T: int = 1000,
    recovery_rate: float = 0.1,
    beta_M: float = 0.6,      # Maximum susceptibility
    batch_size: int = 1000,
    use_contact_matrix: bool = False,
    initial_infected_prop: float = 1e-4,
    SPB_exponent: float = 1.0,
    N_COMPARTMENTS: int = 100
) -> Tuple[StateType, jnp.ndarray]:
    base_params = {
        'recovery_rate': recovery_rate,
        'dT': dT,
        'homophilic_tendency': h,
        'beta_params': (2.0, 2.0),
        'beta_M': beta_M,
        'mu_max': 0.8,  # This will be overwritten by sweep
        'SPB_exponent': SPB_exponent
    }
    
    mask_vals = homogeneous_distribution(mask_max_range["n"], mask_max_range["m"], mask_max_range["M"])
    pol_vals = homogeneous_distribution(pol_range["n"], pol_range["m"], pol_range["M"])
    alpha_vals = pol_to_alpha(pol_vals)
    
    mask_mesh, alpha_mesh = jnp.meshgrid(mask_vals, alpha_vals)
    param_ranges = jnp.stack([mask_mesh.ravel(), alpha_mesh.ravel()], axis=1)
    
    n_steps = int(T / dT)
    
    return batch_sweep(
        simulation_fn=sim_maskSIR_final,
        param_ranges=param_ranges,
        base_params=base_params,
        n_steps=n_steps,
        batch_size=batch_size,
        param_updater=update_mask_pol_params,
        initial_infected_prop=initial_infected_prop,
        N_COMPARTMENTS=N_COMPARTMENTS,
        use_contact_matrix=use_contact_matrix,
        SPB_exponent=SPB_exponent
    )

def sweep_hom_pol_maskSIR(
    h_range: dict = {"m": -2.0, "M": 2.0, "n": 10},
    pol_range: dict = {"m": 0.01, "M": 1, "n": 10},
    dT: float = 1,
    T: int = 1000,
    recovery_rate: float = 0.1,
    beta_M: float = 0.6,      # Maximum susceptibility
    mu_max: float = 0.8,      # Maximum mask-wearing value
    batch_size: int = 1000,
    initial_infected_prop: float = 1e-4,
    SPB_exponent: float = 1.0,
    use_contact_matrix: bool = True,
    N_COMPARTMENTS: int = 100
) -> Tuple[StateType, jnp.ndarray]:
    base_params = {
        'recovery_rate': recovery_rate,
        'dT': dT,
        'beta_M': beta_M,
        'mu_max': mu_max,
        'beta_params': (2.0, 2.0),
        'SPB_exponent': SPB_exponent
    }
    
    h_vals = homogeneous_distribution(h_range["n"], h_range["m"], h_range["M"])
    pol_vals = homogeneous_distribution(pol_range["n"], pol_range["m"], pol_range["M"])
    alpha_vals = pol_to_alpha(pol_vals)
    
    h_mesh, alpha_mesh = jnp.meshgrid(h_vals, alpha_vals)
    param_ranges = jnp.stack([h_mesh.ravel(), alpha_mesh.ravel()], axis=1)
    
    n_steps = int(T / dT)
    
    return batch_sweep(
        simulation_fn=sim_maskSIR_final,
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