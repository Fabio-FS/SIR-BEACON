import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple
from .dynamic import sim_SIRM_final
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

def update_sus_pol_params(params: Dict[str, Any], values: jnp.ndarray) -> Dict[str, Any]:
    """Updates parameters with new susceptibility and alpha values
    
    Args:
        values[0]: max susceptibility value
        values[1]: alpha value (already converted from polarization)
    """
    params['susceptibility_rates'] = (0, values[0])     # min and max susceptibility
    params['beta_params'] = (values[1], values[1])      # both parameters equal
    
    #jax.debug.print("Setting beta_params to: ({}, {})", params['beta_params'][0], params['beta_params'][1])
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

def update_h_susc_params(params: Dict[str, Any], values: jnp.ndarray) -> Dict[str, Any]:
    """Updates parameters with new homophily and susceptibility values
    
    Args:
        params: Base parameter dictionary
        values: Array where:
            values[0]: homophily value (h)
            values[1]: max susceptibility value
    """
    params['homophilic_tendency'] = values[0]
    params['susceptibility_rates'] = (0.0, values[1])
    return params




def sweep_pol_mean_SIRM(
    pol_range: dict = {"m": 0, "M": 1, "n": 100},
    mean_range: dict = {"m": 0, "M": 1, "n": 100},
    h: float = 0,
    dT: float = 1,
    T: int = 1000,
    recovery_rate: float = 0.1,
    susceptibility_rate: float = 0.6,
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
        'susceptibility_rates': (0.0, susceptibility_rate),
        'beta_params': (1.0, 1.0),
        'SPB_exponent': SPB_exponent
    }
    
    pol_vals =  homogeneous_distribution(pol_range["n"], pol_range["m"], pol_range["M"])
    mean_vals = homogeneous_distribution(mean_range["n"], mean_range["m"], mean_range["M"])
    
    pol_mesh, mean_mesh = jnp.meshgrid(pol_vals, mean_vals)
    param_ranges = jnp.stack([pol_mesh.ravel(), mean_mesh.ravel()], axis=1)
    
    n_steps = int(T / dT)
    
    return batch_sweep(
        simulation_fn=sim_SIRM_final,
        param_ranges=param_ranges,
        base_params=base_params,
        n_steps=n_steps,
        batch_size=batch_size,
        param_updater=update_pol_mean_params,
        initial_infected_prop=initial_infected_prop,
        N_COMPARTMENTS = N_COMPARTMENTS,
        use_contact_matrix = use_contact_matrix,
        SPB_exponent = SPB_exponent
    )

def sweep_pol_SPB_SIRM(
    susc_max_range: dict = {"m": 0.2, "M": 0.8, "n": 10},
    pol_range: dict = {"m": 0, "M": 1, "n": 10},
    h: float = 2.0,
    dT: float = 0.25,
    T: int = 1000,
    recovery_rate: float = 0.1,
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
        'susceptibility_rates': (0.0, 0.6),
        'SPB_exponent': SPB_exponent
    }
    
    susc_vals = homogeneous_distribution(susc_max_range["n"], susc_max_range["m"], susc_max_range["M"])
    pol_vals = homogeneous_distribution(pol_range["n"], pol_range["m"], pol_range["M"])
    alpha_vals = pol_to_alpha(pol_vals)
    
    susc_mesh, alpha_mesh = jnp.meshgrid(susc_vals, alpha_vals)
    param_ranges = jnp.stack([susc_mesh.ravel(), alpha_mesh.ravel()], axis=1)
    #jax.debug.print("First few param ranges: {}", param_ranges[:5])
    
    n_steps = int(T / dT)
    
    return batch_sweep(
        simulation_fn=sim_SIRM_final,
        param_ranges=param_ranges,
        base_params=base_params,
        n_steps=n_steps,
        batch_size=batch_size,
        param_updater=update_sus_pol_params,
        initial_infected_prop=initial_infected_prop,
        N_COMPARTMENTS = N_COMPARTMENTS,
        use_contact_matrix = use_contact_matrix,
        SPB_exponent = SPB_exponent
    )

def sweep_hom_pol_SIRM(
    h_range: dict = {"m": -2.0, "M": 2.0, "n": 10},
    pol_range: dict = {"m": 0.01, "M": 1, "n": 10},
    dT: float = 1,
    T: int = 1000,
    recovery_rate: float = 0.1,
    susceptibility_rate: float = 0.6,
    batch_size: int = 1000,
    initial_infected_prop: float = 1e-4,
    SPB_exponent: float = 1.0,
    use_contact_matrix: bool = True,
    N_COMPARTMENTS: int = 100
) -> Tuple[StateType, jnp.ndarray]:
    base_params = {
        'recovery_rate': recovery_rate,
        'dT': dT,
        'susceptibility_rates': (0.0, susceptibility_rate),
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
        simulation_fn=sim_SIRM_final,
        param_ranges=param_ranges,
        base_params=base_params,
        n_steps=n_steps,
        batch_size=batch_size,
        param_updater=update_h_pol_params,
        initial_infected_prop=initial_infected_prop,
        N_COMPARTMENTS = N_COMPARTMENTS,
        use_contact_matrix = use_contact_matrix,
        SPB_exponent = SPB_exponent
    )

def sweep_hom_SPB_SIRM(
    h_range: dict = {"m": -2.0, "M": 2.0, "n": 10},
    susc_range: dict = {"m": 0.2, "M": 0.8, "n": 10},
    dT: float = 1,
    T: int = 1000,
    recovery_rate: float = 0.1,
    alpha: float = 2.0,
    batch_size: int = 1000,
    initial_infected_prop: float = 1e-4,
    SPB_exponent: float = 1.0,
    use_contact_matrix: bool = True,
    N_COMPARTMENTS: int = 100
) -> Tuple[StateType, jnp.ndarray]:
    base_params = {
        'recovery_rate': recovery_rate,
        'dT': dT,
        'beta_params': (alpha, alpha),
        'susceptibility_rates': (0.0, 0.6),
        'SPB_exponent': SPB_exponent
    }
    
    h_vals = homogeneous_distribution(h_range["n"], h_range["m"], h_range["M"])
    susc_vals = homogeneous_distribution(susc_range["n"], susc_range["m"], susc_range["M"])
    
    h_mesh, susc_mesh = jnp.meshgrid(h_vals, susc_vals)
    param_ranges = jnp.stack([h_mesh.ravel(), susc_mesh.ravel()], axis=1)
    
    n_steps = int(T / dT)
    
    return batch_sweep(
        simulation_fn=sim_SIRM_final,
        param_ranges=param_ranges,
        base_params=base_params,
        n_steps=n_steps,
        batch_size=batch_size,
        param_updater=update_h_susc_params,
        initial_infected_prop=initial_infected_prop,
        N_COMPARTMENTS = N_COMPARTMENTS,
        use_contact_matrix = use_contact_matrix,
        SPB_exponent = SPB_exponent
    )
