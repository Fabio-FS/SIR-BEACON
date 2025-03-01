import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple
from .dynamic import sim_maskSIRD_final
from .dynamic import StateType
from ...utils.distributions import pol_mean_to_ab, pol_to_alpha, homogeneous_distribution
from ...utils.batch_sweep import batch_sweep_maskSIRD  # Using the maskSIR_D specific batch sweep

def update_pol_mean_params(params: Dict[str, Any], values: jnp.ndarray) -> Dict[str, Any]:
    """Updates parameters with new alpha and beta values derived from variance and mean"""
    alpha, beta = pol_mean_to_ab(values[0], values[1])
    params['beta_params'] = (alpha, beta)
    return params

def update_mask_pol_params(params: Dict[str, Any], values: jnp.ndarray) -> Dict[str, Any]:
    """Updates parameters with new maximum mask-wearing and alpha values"""
    params['mu_max'] = values[0]           # maximum mask-wearing value
    params['beta_params'] = (values[1], values[1])      # both parameters equal
    return params

def sweep_pol_mask_maskSIRD(
    mask_max_range: dict = {"m": 0, "M": 1, "n": 10},
    pol_range: dict = {"m": 0, "M": 1, "n": 10},
    dT: float = 0.25,
    T: int = 1000,
    recovery_rate: float = 0.1,
    beta_M: float = 0.6,      # Maximum susceptibility
    batch_size: int = 1000,
    initial_infected_prop: float = 1e-4,
    SPB_exponent: float = 1.0,
    N_COMPARTMENTS: int = 100
) -> Tuple[StateType, jnp.ndarray]:
    base_params = {
        'recovery_rate': recovery_rate,
        'dT': dT,
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
    
    return batch_sweep_maskSIRD(
        simulation_fn=sim_maskSIRD_final,
        param_ranges=param_ranges,
        base_params=base_params,
        n_steps=n_steps,
        batch_size=batch_size,
        param_updater=update_mask_pol_params,
        initial_infected_prop=initial_infected_prop,
        N_COMPARTMENTS=N_COMPARTMENTS,
        SPB_exponent=SPB_exponent
    )