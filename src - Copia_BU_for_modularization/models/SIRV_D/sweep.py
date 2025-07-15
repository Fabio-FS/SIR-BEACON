import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple
from ...utils.distributions import pol_to_alpha, homogeneous_distribution
from .dynamic import sim_SIRVD_final, StateType
from ...utils.batch_sweep import batch_sweep_maskSIRD  # Reuse this since structure is the same

def update_vacc_pol_params(params: Dict[str, Any], values: jnp.ndarray) -> Dict[str, Any]:
    """Updates parameters with new vaccination rate and alpha values
    
    Args:
        params: Base parameter dictionary
        values: Array where:
            values[0]: maximum vaccination rate
            values[1]: alpha value (already converted from polarization)
    """
    params['vaccination_rates'] = (0, values[0])     # min and max vaccination rates
    params['beta_params'] = (values[1], values[1])   # both parameters equal for polarization
    return params

def sweep_pol_SPB_SIRVD(
    vacc_max_range: dict = {"m": 0.001, "M": 0.05, "n": 10},
    pol_range: dict = {"m": 0.01, "M": 1, "n": 10},
    dT: float = 0.25,
    T: int = 1000,
    recovery_rate: float = 0.1,
    susceptibility_rate: float = 0.6,
    batch_size: int = 1000,
    initial_infected_prop: float = 1e-4,
    SPB_exponent: float = 1.0,
    N_COMPARTMENTS: int = 100
) -> Tuple[StateType, jnp.ndarray, jnp.ndarray]:
    """Sweep over vaccination rate (SPB) and polarization values for the disconnected SIRV model
    
    Args:
        vacc_max_range: Dictionary defining range for maximum vaccination rate
        pol_range: Dictionary defining range for polarization values
        dT: Time step size
        T: Total simulation time
        recovery_rate: Recovery rate
        susceptibility_rate: Fixed susceptibility rate
        batch_size: Number of simulations to run in parallel
        initial_infected_prop: Initial proportion of infected individuals
        SPB_exponent: Exponent for non-linear behavior adoption
        N_COMPARTMENTS: Number of compartments
        
    Returns:
        Tuple containing:
            - Final states (S, I, R, V)
            - R0 values (always -1)
            - Observable homophily values (always 0)
    """
    base_params = {
        'recovery_rate': recovery_rate,
        'dT': dT,
        'susceptibility_rate': susceptibility_rate,
        'vaccination_rates': (0.0, 0.05),
        'beta_params': (2.0, 2.0),
        'SPB_exponent': SPB_exponent
    }
    
    vacc_vals = homogeneous_distribution(vacc_max_range["n"], vacc_max_range["m"], vacc_max_range["M"])
    pol_vals = homogeneous_distribution(pol_range["n"], pol_range["m"], pol_range["M"])
    alpha_vals = pol_to_alpha(pol_vals)
    
    vacc_mesh, alpha_mesh = jnp.meshgrid(vacc_vals, alpha_vals)
    param_ranges = jnp.stack([vacc_mesh.ravel(), alpha_mesh.ravel()], axis=1)
    
    n_steps = int(T / dT)
    
    return batch_sweep_maskSIRD(
        simulation_fn=sim_SIRVD_final,
        param_ranges=param_ranges,
        base_params=base_params,
        n_steps=n_steps,
        batch_size=batch_size,
        param_updater=update_vacc_pol_params,
        initial_infected_prop=initial_infected_prop,
        N_COMPARTMENTS=N_COMPARTMENTS,
        SPB_exponent=SPB_exponent
    )